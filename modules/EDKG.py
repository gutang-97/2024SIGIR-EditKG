# -*- coding: utf-8 -*-
"""
Created on Wed May 17 00:02:04 2023

@author: comp
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum, scatter_softmax

class SelectAgent(nn.Module):
    def __init__(self, dim,temperature):
        super(SelectAgent,self).__init__()
        self.dim = dim
        # self.ln = nn.LayerNorm(256)
        self.temperature = temperature
        self.select_linear_1 = nn.Linear(self.dim,512)
        self.select_linear_2 = nn.Linear(512,256)
        self.select_linear_3 = nn.Linear(256,1)
        
    def forward(self,x,use_ln=False):
        x = torch.relu(self.select_linear_1(x))
        x =torch.relu(self.select_linear_2(x))
        # if use_ln:
        #     x = self.ln(x) + x
        x = self.select_linear_3(x)
        return x


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_users, n_items,n_entity, n_relation, gamma, max_iter):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.gamma = gamma
        self.max_iter = int(max_iter)
        self.dim = 128

        # self.LN = nn.LayerNorm(64)
        self.activation = nn.LeakyReLU()

    def gumbel_process(self,action_prob, tau=1, dim=-1, hard=True):
        gumbels = (
            -torch.empty_like(action_prob, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)

        gumbels = (action_prob + 0.5 * gumbels)/tau

        y_soft = gumbels.softmax(dim)

        if hard:
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(action_prob, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret

    
    def cosin_smi(self,a,b):
        if len(a.shape) != 3:
            a = a.unsqueeze(1)
            b = b.unsqueeze(1)
        a_norm = a / (torch.norm(a, dim=-1, keepdim=True) + 1e-5)
        b_norm = b / (torch.norm(b, dim=-1, keepdim=True) + 1e-5)
        return torch.matmul(a_norm,b_norm.transpose(1,2))

    def _half_mask(self,a,b):
        ab_sig = torch.sigmoid(torch.rand(a.shape)).to(a.device)
        ab_mask = torch.bernoulli(ab_sig)
        ab_mask_rev = (ab_mask<1).float()
        mask_a = a * ab_mask
        mask_b = b * ab_mask_rev
        return mask_a, mask_b


    def forward(self, user_emb, item_emb,interact_mat):
        
        mat_row = interact_mat._indices()[0, :]  #[689974], interact_mat稀疏矩阵(114737,89196)的行索引
        mat_col = interact_mat._indices()[1, :]  #[689974], interact_mat稀疏矩阵(114737,89196)的列索引
        mat_val = interact_mat._values()         #[689974]，interact_mat稀疏矩阵(114737,89196)中的元素
 
        user_item_mat = torch.sparse.FloatTensor(torch.cat([mat_row, mat_col]).view(2, -1), mat_val,
                                                 size=[self.n_users, self.n_items])   #(114737,30040) --> (n_users,n_items)
        item_user_mat = torch.sparse.FloatTensor(torch.cat([mat_col, mat_row]).view(2, -1), mat_val,
                                                 size=[self.n_items, self.n_users]) 
        user_agg_cf = torch.sparse.mm(user_item_mat, item_emb)
        item_agg_cf = torch.sparse.mm(item_user_mat, user_emb)

        return user_agg_cf, item_agg_cf


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, channel, n_hops, n_users,
                 n_items, n_entities, n_relations, interact_mat, gamma, max_iter,
                 device, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()
        self.channel = channel
        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entity = n_entities
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.device = device
        self.act_func = nn.LeakyReLU()
        self.Select_agent = SelectAgent(channel * 3 ,1)
        self.N_Select_agent = SelectAgent(channel * 3 ,1)

        self.bce_loss = nn.CrossEntropyLoss(label_smoothing=0.2)
        relation_weight = nn.init.xavier_uniform_(torch.empty(n_relations, channel))  # not include interact
        self.relation_weight = nn.Parameter(relation_weight)  # [n_relations - 1, in_channel]
        n_relation_weight = nn.init.xavier_uniform_(torch.empty(n_relations, channel))  # not include interact
        self.n_relation_weight = nn.Parameter(n_relation_weight)  # [n_relations - 1, in_channel]
        
        self.kgc = KGC(n_items=self.n_items,num_ent=self.n_entity,num_rel=self.n_relations,dim=channel)
        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_items=n_items, n_entity= n_entities, n_relation=n_relations, gamma=gamma, max_iter=max_iter).to(self.device))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _update_knowledge(self,two_hpo_kg):
        self.n_edge_index = two_hpo_kg[:,[0,-1]].transpose(1,0)
        self.n_edge_type = two_hpo_kg[:,1]
        
    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _edge_sampling_01(self, edge_index, edge_type, rate=0.5):
        n_edges = edge_index.shape[1]
        m = np.random.choice([0, 1], size=n_edges, p=[0.0, 1.0])
        return m

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()                                       #noise_shape：interact_mat:(114737,89196)中的非零元素的数量，为1380510

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)        #随机生成与noise_shape数量一样的tensor:(1380510,)，再加上random_tensor(0.5)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)   #(1380510,), random_tensor中大于1的部分mask为True，小于1的部分mask为False.
        i = x._indices()                                             #(2,1380510)-->非0元素的行和列
        v = x._values()                                              #(1380510,) -->非0元素的值

        i = i[:, dropout_mask]                                       #(2,689974), 丢掉了(2,1380510)中mask为Fasle的部分。
        v = v[dropout_mask]                                          #(689974,), 丢掉了(1380510,)中mask为Fasle的部分。

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)   #(114737,89196)，形状与interact_mat一样，但是矩阵里面的非0元素少了0.5倍。
        # return out * (1. / (1 - rate))                               #对每个元素放大2倍，比如某一行有4个数，都是0.25，变为2个数后每个数自然要变为0.5。
        return out

    def Gumbel_process(self,action_prob, tau=1, dim=-1, hard=True):
        gumbels = (
            -torch.empty_like(action_prob, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        
        gumbels = (action_prob +  gumbels)/tau

        # y_soft = gumbels.softmax(dim)
        y_soft = torch.sigmoid(gumbels)
        y_hard = (y_soft > 0.1).float()

        return y_soft,y_hard
    
    def Dnoise_KG(self,edge_index, edge_type, entity_emb,relation_weight,is_gumble=True,new=False):
        head, tail = edge_index
        head_emb = entity_emb[head]
        tail_emb = entity_emb[tail]
        rel_emb = relation_weight[edge_type]
        
        h_r_t_emb = F.normalize(torch.cat([head_emb,rel_emb,tail_emb],dim=-1))

        if new:
            action_prob = self.N_Select_agent(h_r_t_emb)
        else:
            action_prob = self.Select_agent(h_r_t_emb)

        dims=-1
        if is_gumble:
            action_soft, action_hard= self.Gumbel_process(action_prob, tau=1, dim=dims, hard=True)
            # action = actions[:,1]
        else:
            action_soft = torch.sigmoid(action_prob)
            action_hard = (action_soft > 0.1).float()

        print("keep rate: ",action_hard.sum()/action_hard.shape[0])
        return action_soft, action_hard

    def KG_forward(self, entity_emb, edge_index, edge_type,
                   relation_weight, KG_drop_soft, KG_drop_hard):
        
        n_entities = entity_emb.shape[0] 
        head, tail = edge_index 

        head_emb = entity_emb[head]
        tail_emb = entity_emb[head]
        rel_emb  = relation_weight[edge_type]
        
        KG_score = KG_drop_soft * KG_drop_hard
        
        neb_kg_emb = (tail_emb + rel_emb) * KG_score
        entity_agg = scatter_sum(src=neb_kg_emb , index=head, dim_size=n_entities, dim=0)       
        score_agg = scatter_sum(src=KG_drop_hard , index=head, dim_size=n_entities, dim=0) 
        entity_agg = entity_agg / (score_agg+1e-9)

        score_mask = (score_agg < 1).float()
        # entity_agg = entity_agg + score_mask * entity_emb
        return entity_agg, score_mask


    
    def split_kg(self,edge_index,edge_type,kg_mask_size=512):
        # topk_mask = np.zeros(edge_index.shape[0], dtype=bool)
        # topk_mask[topk_egde_id] = True
        # add another group of random mask
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(
            n_edges, size=kg_mask_size, replace=False)
        random_mask = np.zeros(edge_index.shape[1], dtype=bool)
        random_mask[random_indices] = True

        mask_edge_index = edge_index[:, random_mask]
        mask_edge_type = edge_type[random_mask]
        
        retain_edge_index = edge_index[:,~random_mask]
        retain_edge_type = edge_type[~random_mask]
        
        return retain_edge_index, retain_edge_type, mask_edge_index, mask_edge_type
    
    def create_mae_loss(self, node_pair_emb, masked_edge_emb=None):
        head_embs, tail_embs = node_pair_emb[:, 0, :], node_pair_emb[:, 1, :]
        if masked_edge_emb is not None:
            pos1 = tail_embs * masked_edge_emb
        else:
            pos1 = tail_embs
        # scores = (pos1 - head_embs).sum(dim=1).abs().mean(dim=0)
        scores = - \
            torch.log(torch.sigmoid(torch.mul(pos1, head_embs).sum(1))).mean()
        return scores
    
    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        return mf_loss
    
    def create_bce_loss(self,head_emb,rel_emb,target_id,all_embed):
        merge = head_emb + rel_emb
        score = torch.matmul(merge, all_embed.transpose(1,0))
        bce_loss = self.bce_loss(score, target_id)
        return bce_loss
    
    def _guassian_kernel(self,kg_drb,cf_drb,kernel_mul=2.0, kernel_num=5):
        n_samples = int(kg_drb.size()[0])+int(cf_drb.size()[0])
        total = torch.cat([kg_drb, cf_drb], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

        L2_distance = ((total0-total1)**2).sum(2) 
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def _cal_mmd(self,kg_drb, cf_drb, kernel_mul=2.0, kernel_num=5):
        batch_size = int(kg_drb.size()[0]) 
        kernels = self._guassian_kernel(kg_drb, cf_drb,
            kernel_mul=kernel_mul, kernel_num=kernel_num)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        mmd_loss = torch.mean(XX + YY - XY -YX)
        return mmd_loss
    

    
    def forward(self, all_embed, all_embed_cf, edge_index, edge_type, n_edge_index, n_edge_type, 
                interact_mat, mess_dropout=True, node_dropout=False,gumbel=True):
        #KG_DropEdge_para:(3,558310) (由0，1组成)
        """node dropout"""

        if node_dropout:
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)          #(690255,2);(690255,) 
                                                                   #(89196,64)
                                                                   
        mat_row = interact_mat._indices()[0, :]  #[689974], interact_mat稀疏矩阵(114737,89196)的行索引
        mat_col = interact_mat._indices()[1, :]  #[689974], interact_mat稀疏矩阵(114737,89196)的列索引
        mat_val = interact_mat._values()         #[689974]，interact_mat稀疏矩阵(114737,89196)中的元素

        user_item_mat = torch.sparse.FloatTensor(torch.cat([mat_row, mat_col]).view(2, -1), mat_val,
                                                 size=[self.n_users, self.n_items])   #(114737,30040) --> (n_users,n_items)
        item_user_mat = torch.sparse.FloatTensor(torch.cat([mat_col, mat_row]).view(2, -1), mat_val,
                                                 size=[self.n_items, self.n_users]) 
        # user_embeds = all_embed[:self.n_users][:,:self.channel]
        user_embeds = all_embed[:self.n_users]
        item_embed = all_embed[self.n_users:self.n_users + self.n_items][:,:self.channel]
        entity_emb = all_embed[self.n_users:][:,self.channel:self.channel * 2]
        n_entity_emb = all_embed[self.n_users:][:,self.channel*2:]

        '''KG'''
        KG_drop_soft, KG_drop_hard = self.Dnoise_KG(edge_index, edge_type, entity_emb, self.relation_weight, is_gumble=gumbel)
        N_KG_drop_soft, N_KG_drop_hard = self.Dnoise_KG(self.n_edge_index, self.n_edge_type, n_entity_emb, self.n_relation_weight, is_gumble=gumbel,new=True)

        '''calculate mmd loss'''
        kg_trip = torch.cat([self.n_edge_index[0].unsqueeze(0),self.n_edge_type.unsqueeze(0),self.n_edge_index[1].unsqueeze(0)],dim=0).transpose(1,0)
        mmd_batch = torch.randint(low=0,high=len(kg_trip),size=(int(4096 * 2),)).to(kg_trip.device)   ##MIND:4096 * 2 others:4096
        kg_bc_trip = kg_trip[mmd_batch]
        bc_n_kg_drop_soft = N_KG_drop_soft[mmd_batch]
        bc_kgc_soft = self.kgc(kg_bc_trip,eval=True,cf_train=True)
        bc_kgc_hard = (bc_kgc_soft > 0.1).float().sum()
        print("bc_kgc_hard",bc_kgc_hard)
        # print(bc_n_kg_drop_soft[:100])
        bc_kgc_soft = bc_kgc_soft.detach()
        mmd_loss = self._cal_mmd(bc_kgc_soft, bc_n_kg_drop_soft) 
        print("mmd loss: ",mmd_loss.item())
        
        entity_emb_res = entity_emb[:self.n_items]
        n_entity_emb_res = n_entity_emb[:self.n_items]
        for i in range(len(self.convs)):
            entity_emb, ent_mask = self.KG_forward(entity_emb,edge_index,edge_type,self.relation_weight,KG_drop_soft, KG_drop_hard)
            entity_emb_res = entity_emb_res + F.normalize(entity_emb[:self.n_items])
            n_entity_emb, n_ent_mask = self.KG_forward(n_entity_emb,self.n_edge_index,self.n_edge_type,
                                                       self.n_relation_weight,N_KG_drop_soft, N_KG_drop_hard)
            n_entity_emb_res = n_entity_emb_res + F.normalize(n_entity_emb[:self.n_items])

        # KG_drop_hard = None
        '''KG'''
        item_embeds = torch.cat([item_embed,entity_emb_res,n_entity_emb_res],dim=-1)

        user_embed_res = user_embeds
        item_embed_res = item_embeds
        for i in range(len(self.convs)-1):
            user_embeds,item_embeds = self.convs[i](user_embeds, item_embeds,self.interact_mat)
            item_embed_res = item_embed_res + F.normalize(item_embeds)
            user_embed_res = user_embed_res + F.normalize(user_embeds)
            # user_embeds = F.normalize(user_embeds)
            # item_embeds = F.normalize(item_embeds)
            # user_embed_res = user_embed_res + F.dropout(user_embeds,p=0.0,training=self.training)
            # item_embed_res = item_embed_res + F.dropout(item_embeds,p=0.0,training=self.training)
        
        # user_agg_cf = torch.sparse.mm(user_item_mat, item_embed)
        # item_agg_cf =  torch.sparse.mm(item_user_mat, user_embed)
        # user_agg_kg = torch.sparse.mm(user_item_mat, entity_emb_res)
        # user_agg_nkg = torch.sparse.mm(user_item_mat, n_entity_emb_res)
        
        
        # user_emb_all_res = [user_embed_res, user_agg_kg, user_agg_nkg]
        # item_emb_all_res = [item_embed_res, entity_emb_res, n_entity_emb_res]
        # user_emb_all_res = torch.cat([user_embed + user_agg_cf,user_agg_kg,user_agg_nkg],dim=-1)
        # item_emb_all_res = torch.cat([item_embed,entity_emb_res,n_entity_emb_res],dim=-1)
        # item_emb_all_res = item_embed
        
        bce_loss = 0
        return user_embed_res, item_embed_res, KG_drop_hard, mmd_loss



class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, ui_sp_graph,item_rel_mask):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.margin_ccl = args_config.margin
        self.num_neg_sample = args_config.num_neg_sample
        
        self.gamma = args_config.gamma
        self.max_iter = args_config.max_iter
        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.loss_f = args_config.loss_f
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.item_rel_mask = torch.FloatTensor(item_rel_mask).to(self.device)
        self.ui_sp_graph = ui_sp_graph

        self.edge_index, self.edge_type = self._get_edges(graph)
        
        self.n_edge_index = None
        self.n_edge_type = None

        self.cet_loss = nn.CrossEntropyLoss(label_smoothing=0.85)
        # self.ranking_loss = nn.MarginRankingLoss(margin=1)
        self._init_weight()
        self._init_loss_function()

        self.gcn = self._init_model()

        
        
    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size * 3))
        self.all_embed = nn.Parameter(self.all_embed)

        # self.all_embed_cf = initializer(torch.empty(self.n_users + self.n_items, self.emb_size))
        # self.all_embed_cf = nn.Parameter(self.all_embed_cf)
        self.all_embed_cf = None
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.ui_sp_graph).to(self.device)   #sparse_tensor: (114737,89196) user-item的交互矩阵，行是user,列是item,只有前30040(n_items)有值。

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)


    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_items=self.n_items,
                         n_entities=self.n_entities,
                         n_relations=self.n_relations,
                         interact_mat=self.interact_mat,
                         gamma=self.gamma,
                         max_iter=self.max_iter,
                         device=self.device,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)



    def _init_loss_function(self):
        if self.loss_f == "inner_bpr":
            self.loss = self.create_inner_bpr_loss
        elif self.loss_f == 'contrastive_loss':
            self.loss = self.create_contrastive_loss
        else:
            raise NotImplementedError

    def L2_norm(self,hidden,k=1):
        hidden_norm = torch.norm(hidden,p=2,dim=-1,keepdim=True)
        out_hidden = k * torch.div(hidden,hidden_norm+1e-8)
        return out_hidden

    def _contrastive_loss(self,user_kg,n_user_kg,pos_user_id, tau=1,small_batch = 256):
        if small_batch:
            small_id = torch.randint(low=0, high=pos_user_id.shape[0], size=(small_batch,)).to(user_kg.device)
            pos_user_id = pos_user_id[small_id]
            
        pos_user_kge = user_kg[pos_user_id]
        pos_n_user_kge = n_user_kg[pos_user_id]

        pos = torch.eye(pos_user_kge.shape[0]).to(pos_user_kge.device)
        z1_norm = torch.norm(pos_user_kge, dim=-1, keepdim=True)
        z2_norm = torch.norm(pos_n_user_kge, dim=-1, keepdim=True)
        dot_numerator = torch.mm(pos_user_kge, pos_n_user_kge.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / (dot_denominator + 1e-5) / tau)
        
        smi_sum = torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-5

        sim_matrix = sim_matrix/smi_sum
        assert sim_matrix.size(0) == sim_matrix.size(1)
        cl_loss = -torch.log(sim_matrix.mul(pos).sum(dim=-1)).mean()

        return cl_loss


    def gcn_forword(self, user, pos_item):
        user_all_emb, item_all_emb, KG_drop_hard,mmd_loss  = self.gcn(self.all_embed,
                                                            self.all_embed_cf,
                                                            self.edge_index,
                                                            self.edge_type,
                                                            self.n_edge_index,
                                                            self.n_edge_type,
                                                            self.interact_mat,
                                                            mess_dropout=self.mess_dropout,
                                                            node_dropout=self.node_dropout,
                                                            gumbel = True )
        # user_kg = user_all_emb[1]
        # user_n_kg = user_all_emb[2]
        # cl_loss = self._contrastive_loss(user_kg,user_n_kg,user,tau=1,small_batch = 256)
        # print("cl loss: ",cl_loss.item())
        # user_all_emb = torch.cat(user_all_emb,dim=-1)
        # item_all_emb = torch.cat(item_all_emb,dim=-1)
        user_emb = user_all_emb[user]
        score = torch.matmul(user_emb, item_all_emb.transpose(1,0))
        cet_loss = self.cet_loss(score,pos_item)
        

        pos_emb = item_all_emb[pos_item]
        regularizer = (torch.norm(user_emb) ** 2
                       + torch.norm(pos_emb) ** 2)
 
        return  cet_loss, mmd_loss 
    


    def forward(self,batch=None,mode="cf"):
        if mode == "cf":
            user = batch['users']                                                                          #(4096,)
            pos_item = batch['pos_items']                                                                  #(4096,)

            loss_network = self.gcn_forword(user, pos_item)
            return loss_network
        
        else:
            kgc_loss = self.gcn.kgc(batch)
            return kgc_loss


    def generate(self,for_kgc=False):
        # user_emb = self.all_embed[:self.n_users, :]
        # entity_emb = self.all_embed[self.n_users:, :]
        user_all_emb, item_all_emb,KG_drop_hard,kg_loss = self.gcn( self.all_embed,
                                                            self.all_embed_cf,
                                                            self.edge_index,
                                                            self.edge_type,
                                                            self.n_edge_index,
                                                            self.n_edge_type,
                                                            self.interact_mat,
                                                            mess_dropout=False,
                                                            node_dropout=False,
                                                            gumbel = False )
        # user_all_emb = torch.cat(user_all_emb,dim=-1)
        # item_all_emb = torch.cat(item_all_emb,dim=-1)
        
        item_pred_emb = item_all_emb 
        user_pred_emb = user_all_emb 
        if for_kgc:
            return item_pred_emb, KG_drop_hard
        else:
            return item_pred_emb, user_pred_emb


    def rating(self, u_g_embeddings, i_g_embeddings,type="bpr"):
        if type == "bpr":
            return torch.matmul(u_g_embeddings, i_g_embeddings.t()).detach().cpu()

        else:
            return torch.cosine_similarity(u_g_embeddings[:, :self.emb_size].unsqueeze(1),
                                           i_g_embeddings[:, :self.emb_size].unsqueeze(0), dim=2).detach().cpu() + \
                   torch.cosine_similarity(u_g_embeddings[:, self.emb_size:].unsqueeze(1),
                                           i_g_embeddings[:, self.emb_size:].unsqueeze(0), dim=2).detach().cpu()


    def create_contrastive_loss(self, u_e, pos_e, neg_e,loss_weight):
        batch_size = u_e.shape[0]

        u_e = F.normalize(u_e)
        pos_e = F.normalize(pos_e)
        neg_e = F.normalize(neg_e)

        ui_pos_loss1 = torch.relu(1 - torch.cosine_similarity(u_e, pos_e, dim=1))

        users_batch = torch.repeat_interleave(u_e, self.num_neg_sample, dim=0)

        ui_neg1 = torch.relu(torch.cosine_similarity(users_batch, neg_e, dim=1) - self.margin_ccl)
        ui_neg1 = ui_neg1.view(batch_size, -1)
        x = ui_neg1 > 0
        ui_neg_loss1 = torch.sum(ui_neg1, dim=-1) / (torch.sum(x, dim=-1) + 1e-5)

        loss = (ui_pos_loss1 + ui_neg_loss1)

        return loss.mean()


    def create_inner_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        cf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return cf_loss + emb_loss
    
    
    
    
    
class KGC(nn.Module):
    def __init__(self, n_items,num_ent, num_rel, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
        super(KGC, self).__init__()
        self.n_items = n_items
        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.linear_1 = nn.Linear(self.dim * 2,512)
        # self.linear_2 = nn.Linear(1024,1024)
        # self.linear_2 = nn.Linear(1024,512)
        self.linear_2 = nn.Linear(512,256)
        # self.ln = nn.LayerNorm(256)
        self.linear_pre = nn.Linear(256,1)
                

        self.ent_embeddings = nn.Embedding(self.num_ent,self.dim)
        self.rel_embeddings = nn.Embedding(self.num_rel, self.dim)

        self.loss_F = nn.MarginRankingLoss(self.margin, reduction="mean")
        # self.bce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.bce_loss = nn.BCELoss(reduction="mean")
    def __parameter_init(self,normalize=False):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        if normalize:
            self.normalization_rel_embedding()
            self.normalization_ent_embedding()

    def normalization_ent_embedding(self):
        norm = self.ent_embeddings.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        self.ent_embeddings.weight.data.copy_(torch.from_numpy(norm))

    def normalization_rel_embedding(self):
        norm = self.rel_embeddings.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        self.rel_embeddings.weight.data.copy_(torch.from_numpy(norm))
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
    
    def _distance(self, h, t, r,neg=False):
        if neg:
            score = (h + r).unsqueeze(1) - t
            score = torch.norm(score, p=self.p_norm, dim=-1).mean(dim=1)
        else:
            score = (h + r) - t
            score = torch.norm(score, p=self.p_norm, dim=1)
        return score

    def forward(self, data, eval=False,rate=0.5,cf_train=False):

        if eval:
            if cf_train:
                batch_triple = data
            else:
                batch_triple = data["hr_pair"]
            
            batch_h = batch_triple[:,0]
            batch_r = batch_triple[:,1]
            batch_t = batch_triple[:,2]
            # batch_label = batch_triple[:,-1]

            h = self.ent_embeddings(batch_h)
            r = self.rel_embeddings(batch_r)
            t = self.ent_embeddings(batch_t)
            
            # score = torch.sigmoid(torch.matmul(h,(r * t)))
            h_r_t_emb = F.normalize(torch.cat([h,r*t],dim=-1))
            h_r_t_emb = torch.relu(self.linear_1(h_r_t_emb))
            h_r_t_emb = torch.relu(self.linear_2(h_r_t_emb))
            # h_r_t_emb = torch.relu(self.linear_3(h_r_t_emb))
            # h_r_t_emb = torch.relu(self.linear_4(h_r_t_emb))
            score = torch.sigmoid(self.linear_pre(h_r_t_emb))
            # return (score>=rate).squeeze(-1).float()
            return score
        
        batch_triple = data["hr_pair"]
        
        batch_h = batch_triple[:,0]
        batch_r = batch_triple[:,1]
        batch_t = batch_triple[:,2]
        batch_label = batch_triple[:,-1]

        h = self.ent_embeddings(batch_h)
        r = self.rel_embeddings(batch_r)
        t = self.ent_embeddings(batch_t)
        h_r_t_emb = F.normalize(torch.cat([h,r*t],dim=-1))
        h_r_t_emb = torch.relu(self.linear_1(h_r_t_emb))
        h_r_t_emb = torch.relu(self.linear_2(h_r_t_emb))
        # h_r_t_emb = torch.relu(self.linear_3(h_r_t_emb))
        # h_r_t_emb = torch.relu(self.linear_4(h_r_t_emb))
        score = torch.sigmoid(self.linear_pre(h_r_t_emb))

        loss = self.bce_loss(score.squeeze(-1), batch_label.float())
 
        return loss

    def regularization(self, data):

        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings[batch_h]
        t = self.ent_embeddings[batch_t]
        r = self.rel_embeddings[batch_r]
        regul = (torch.mean(h ** 2) + 
                    torch.mean(t ** 2) + 
                    torch.mean(r ** 2)) / 3
        return regul
