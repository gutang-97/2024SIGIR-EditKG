# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:45:19 2023

@author: comp
"""

import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import pickle as pkl

import random
from time import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)


def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    inter_mat = np.array(inter_mat)
    print("max before: ",max(inter_mat[:,1]))
    inter_mat[:,1]+=1
    print("max after: ",max(inter_mat[:,1]))
    return inter_mat


def remap_item(train_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))





def read_triplets(file_name):
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    if args.inverse_r:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # consider an additional relations --- 'interact'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider a additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()
    triplets[:,0]+=1
    triplets[:,2]+=1
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    # return triplets
    return np.unique(triplets, axis=0)


def generate_polluted_cf_data(train_cf, rate):
    index = np.arange(len(train_cf))
    np.random.shuffle(index)
    train_cf = train_cf[index]

    n_noise = int(len(train_cf) * rate)
    train_cf_noise = train_cf[:n_noise]
    train_cf_ori = train_cf[n_noise:]

    train_total = []
    for u, i in train_cf_noise:
        while 1:
            n = np.random.randint(low=0, high=n_items, size=1)[0]
            if n not in train_user_set[u]:
                train_total.append([u, n])
                break
    train_total = np.vstack((np.array(train_total), train_cf_ori))

    train_dict = defaultdict(list)
    for u, i in train_total:
        train_dict[int(u)].append(int(i))

    f = open('./data/{}/train_noise_{}.txt'.format(args.dataset, rate), 'w')
    for key, val in train_dict.items():
        val = [key] + val
        val = ' '.join(str(x) for x in val)
        val = val + '\n'
        f.write(val)
    f.close()


def generate_polluted_kg_data(file_name, rate):
    triplets_np = np.loadtxt(file_name, dtype=np.int32)
    triplets_np = np.unique(triplets_np, axis=0)

    tri_dict = defaultdict(list)
    for h, r, t in triplets_np:
        tri_dict[int(h)].append(int(t))

    index = np.arange(len(triplets_np))
    np.random.shuffle(index)
    triplets_np = triplets_np[index]

    n_noise = int(len(triplets_np) * rate)
    triplets_np_noise = triplets_np[:n_noise]
    triplets_np_ori = triplets_np[n_noise:]

    triplets_np_total = []
    for h, r, t in triplets_np_noise:
        while 1:
            n = np.random.randint(low=0, high=n_entities, size=1)[0]
            if n not in tri_dict[h]:
                triplets_np_total.append([h, r, n])
                break
    triplets_np_total = np.vstack((np.array(triplets_np_total), triplets_np_ori))

    f = open('./data/{}/kg_noise_{}.txt'.format(args.dataset, rate), 'w')
    for h, r, t in triplets_np_total:
        f.write(str(h) + ' ' + str(r) + ' ' + str(t) + '\n')
    f.close()

def build_graph(train_data, triplets):
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)

    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id])

    return ckg_graph, rd


def build_sparse_graph(relation_dict):
    # def _bi_norm_lap(adj):
    #     # D^{-1/2}AD^{-1/2}
    #     rowsum = np.array(adj.sum(1))

    #     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    #     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    #     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    #     # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    #     bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    #     return bi_lap.tocoo()

    # def _si_norm_lap(adj):
    #     # D^{-1}A
    #     rowsum = np.array(adj.sum(1))

    #     d_inv = np.power(rowsum, -1).flatten()
    #     d_inv[np.isinf(d_inv)] = 0.
    #     d_mat_inv = sp.diags(d_inv)

    #     norm_adj = d_mat_inv.dot(adj)
    #     return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):                                                    #relation_dict 一共103种关系
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:                                                                          #是user-item的交互
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)        #因为user和item都是从0开始编码，为了能够在稀疏矩阵中存储，所以item在矩阵表示为列，所以列标从n_user开始
            vals = [1.] * len(cf)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))        #n_nodes：203933， 但是这个adj中只有前(n_users+n_items) * (n_users+n_items)--> (114737+30040=144777) * (114737+30040=144777)中有值
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes)) #由于np_mat里都是entity,所以这个里面的np_mat最大编码数为:0-->n_entity=89196, 所以这个adj只有前 89196行*89196列有值。
        adj_mat_list.append(adj)
    ui_mat_list = adj_mat_list[0].tocsr()[:n_users, n_users:].tocoo()                            #变成了n_user * n_entities大小了:(114737, 89196) 但是只有前(n_user,n_items)-->(114737,30040)有值.
    return ui_mat_list, adj_mat_list


def build_item_onestep_relation(triplets):
    item_neb_rel_dict = defaultdict(list)
    item_neb_ent_dict = defaultdict(list)
    max_len = 0
    min_len = 10000
    avg_len = 0
    for i in tqdm(range(1,n_items+1),desc="buliding item relation dict ...."):
        temp_trip = triplets[triplets[:,0]==i]
        item_neb_rel_dict[i] = temp_trip[:,1]
        item_neb_ent_dict[i] = temp_trip[:,2]
        
        if max_len < len(item_neb_rel_dict[i]):max_len= len(item_neb_rel_dict[i])
        if min_len > len(item_neb_rel_dict[i]):min_len = len(item_neb_rel_dict[i])
        avg_len+=len(item_neb_rel_dict[i])
        
                                                                                                                                                                                                                  
    print("KG onehop relation maxlen: ",max_len)
    print("KG onehop relation minlen: ",min_len)
    print("KG onehop relation avglen: ",avg_len/len(item_neb_rel_dict))
    def padding_masking(v,max_len):
        mask = [1] * len(v)
        if len(v) < max_len:return (v + [0] * (max_len-len(v)), mask+[0] * (max_len-len(v)))
        else: return (v[:max_len],mask[:max_len])
    
    item_rel_dict = defaultdict(list)
    item_rel_dict = {k:(v,np.asarray([1]*len(v))) for k, v in item_neb_rel_dict.items()}
    
    item_ent_dict = defaultdict(list)
    item_ent_dict = {k:(v,np.asarray([1]*len(v))) for k, v in item_neb_ent_dict.items()}
    return item_rel_dict, item_ent_dict



def load_data(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    remap_item(train_cf, test_cf)

    print('combinating train_cf and kg data ...')
    triplets = read_triplets(directory + 'kg_final.txt')
    # item_rel_dict, item_ent_dict = build_item_onestep_relation(triplets)
    # pkl.dump((item_rel_dict,item_ent_dict),open("item_rel_ent.pkl","wb"))
    item_rel_dict,item_ent_dict = pkl.load(open("item_rel_ent.pkl","rb"))
    print("loaded item rel ent dict")


    print('building the graph ...')
    graph, relation_dict = build_graph(train_cf, triplets)

    ui_sparse_graph, all_sparse_graph = build_sparse_graph(relation_dict)  
    
    
    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items) ,
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict, n_params, graph,\
           ui_sparse_graph, all_sparse_graph,item_rel_dict,item_ent_dict
           
           
           