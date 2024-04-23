import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class KGR(nn.Module):
    def __init__(self, n_items,num_ent, num_rel, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
        super(KGR, self).__init__()
        self.n_items = n_items
        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.linear_1 = nn.Linear(self.dim * 2,1024)
        # self.linear_2 = nn.Linear(1024,1024)
        self.linear_2 = nn.Linear(1024,512)
        self.linear_3 = nn.Linear(512,256)
        self.ln = nn.LayerNorm(256)
        self.linear_pre = nn.Linear(256,1)
                
        # initializer = nn.init.xavier_uniform_
        # self.ent_embeddings = initializer(torch.empty(self.num_ent, self.dim))
        # self.ent_embeddings = nn.Parameter(self.ent_embeddings)
        
        self.ent_embeddings = nn.Embedding(self.num_ent,self.dim)
        self.rel_embeddings = nn.Embedding(self.num_rel, self.dim)
        # self.rel_embeddings = initializer(torch.empty(self.num_rel, self.dim))
        # self.rel_embeddings = nn.Parameter(self.rel_embeddings)
        # self.__parameter_init()
        
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
    
    def _update_kgc_ui_graph(self,kgc_ui_graph):
        self.kgc_ui_graph = self._convert_sp_mat_to_sp_tensor(kgc_ui_graph).cuda()

    def _distance(self, h, t, r,neg=False):
        if neg:
            score = (h + r).unsqueeze(1) - t
            score = torch.norm(score, p=self.p_norm, dim=-1).mean(dim=1)
        else:
            score = (h + r) - t
            score = torch.norm(score, p=self.p_norm, dim=1)
        return score

    def _kgcn(self):
        item_embeddings = torch.tanh(torch.sparse.mm(self.kgc_ui_graph,self.ent_embeddings[:self.n_items])) 
        ie_embeddings = item_embeddings + self.ent_embeddings[:self.n_items]
        ent_embeddings = torch.cat([ie_embeddings,self.ent_embeddings[self.n_items:]],dim=0)
        return ent_embeddings,self.rel_embeddings
    
    def forward(self, data, eval=False,rate=0.5):
        # ent_embeddings, rel_embeddings = self._kgcn()
        
        if eval:
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
            h_r_t_emb = torch.relu(self.linear_3(h_r_t_emb))
            # h_r_t_emb = torch.relu(self.linear_4(h_r_t_emb))
            score = torch.sigmoid(self.linear_pre(h_r_t_emb))
            return (score>=rate).squeeze(-1).float()
        
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
        h_r_t_emb = torch.relu(self.linear_3(h_r_t_emb))
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
