# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:45:19 2023

@author: comp
"""

import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp

import random
from time import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import pickle as pkl
import os
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

    return np.array(inter_mat)

 
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

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    def _get_kg_dict(triplets):
        kg_dict = defaultdict(list)
        for h,r,t in triplets:
            kg_dict[h].append((r,t))
        return kg_dict
    kg_dict = _get_kg_dict(triplets)
    return triplets, kg_dict


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


    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])


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

    #     bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    #     # bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    #     return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []

    for r_id in tqdm(relation_dict.keys()):                                                    
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:                                                                          
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)        
            vals = [1.] * len(cf)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))        
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes)) 
        adj_mat_list.append(adj)
    ui_mat_list = adj_mat_list[0].tocsr()[:n_users, n_users:n_users + n_items].tocoo()                           
    # return _si_norm_lap(ui_mat_list), adj_mat_list
    return ui_mat_list, adj_mat_list

def build_kg_set(triplets):
    if os.path.exists("item_rel_mask_rev.pkl"):
        item_rel_mask_rev = pkl.load(open("item_rel_mask_rev.pkl","rb"))
    else:
        item_rel_mask = np.zeros((n_items,n_relations))
        for item in tqdm(range(n_items)):
            item_rel = triplets[triplets[:,0]==item][:,1]
            item_rel = np.unique(item_rel)
            item_rel_map = item_rel_mask[item]
            item_rel_map[item_rel] = 1
            if sum(item_rel_map) == 0:
                pass
            item_rel_mask[item] = item_rel_map
            # item_rel_set[item] = item_rel.tolist()
        item_rel_mask_rev = (~(item_rel_mask>0)).astype("float")
        item_rel_mask_rev[:,0] = 0
        pkl.dump(item_rel_mask_rev,open("item_rel_mask_rev.pkl","wb"))
    return item_rel_mask_rev

def static_kg(triplets):
    head = triplets[:,0]
    tail = triplets[:,2]
    cnt_both_non = 0
    cnt_both_in = 0
    cnt_one_in = 0
    for i in range(len(head)):
        if head[i]>n_items and tail[i]>n_items:
            cnt_both_non+=1
        elif head[i]<n_items and tail[i]<n_items:
            cnt_both_in+=1
        else:
            cnt_one_in += 1
    # cnt_item = 0
    # for item in tqdm(range(n_entities)):
    #     if item not in head:
    #         cnt_item+=1

def load_data(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'


    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    remap_item(train_cf, test_cf)


    # generate_polluted_cf_data(train_cf, rate=0.2)
    # exit()
 

    triplets,kg_dict = read_triplets(directory + 'kg_final.txt')
    static_kg(triplets)
    item_rel_mask = build_kg_set(triplets)

    graph, relation_dict = build_graph(train_cf, triplets)

    ui_sparse_graph, all_sparse_graph = build_sparse_graph(relation_dict)  
    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict, n_params, graph, ui_sparse_graph, all_sparse_graph,item_rel_mask,triplets, kg_dict
