# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:45:11 2023

@author: comp
"""
import os

import sys
import math
import random
import torch
import itertools
import numpy as np
import pickle as pkl
from math import log
from tqdm import tqdm
from time import time 
import multiprocessing
import scipy.sparse as sp
from collections import Counter, defaultdict

from utils.parser import parse_args
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score
from utils.Speed_data_loader_add_kg import load_data
from sklearn.metrics.pairwise import cosine_similarity

from modules.KGR_model import KGR
from modules.EDKG import Recommender
from modules.pcgrad import PCGrad

from utils.evaluate import test
from utils.helper import early_stopping, _generate_candi_kg, _cal_npmi


cores = multiprocessing.cpu_count()
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_data(train_entity_pairs, train_user_set):

    feed_dict = {}
    entity_pairs = train_entity_pairs
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]

    return feed_dict

def _mulp_neg(triplet):
    h,r,t = triplet
    rand_value = random.random()
    while True:
        if rand_value < 0.33:
            new_rel = random.randint(1,n_relations-1)
            if new_rel != r:
                new_triplet = [h,new_rel,t]
                break
        elif rand_value>=0.33 and rand_value>0.66:
            new_head = random.randint(0,n_entities-1)
            if new_head != h:
                new_triplet = [new_head,r,t]
                break
        else:
            new_t = random.randint(n_items,n_entities-1)
            if new_t != t:
                new_triplet = [h,r,new_t]
                break
    return new_triplet

def _get_KGC_neg_data(triplets,neg_times = 4):
    def negative_sampling(triplets):
        pool = multiprocessing.Pool(cores)
        print("process neg items...")
        # neg_items = pool.map(get_neg_one, user_item.cpu().numpy()[:, 0])
        neg_triplets = list(tqdm(iterable=pool.imap(_mulp_neg,triplets.tolist()),total=len(triplets)))
        pool.close()
        return np.array(neg_triplets)
    all_neg = []
    for i in range(neg_times):
        neg_triplets = negative_sampling(triplets)
        all_neg.append(neg_triplets)
    return np.unique(np.concatenate(all_neg,axis=0),axis=0)


def train_kgr_model(model,kgr_optimizer, triplets, kg_mask=[],epochs=2,threshold=0.5):
    # neg_kg_data = torch.LongTensor(neg_kg_data[index])
    print("triplets: ",triplets.shape)
    if kg_mask!=[]:
        triplets = triplets[kg_mask.reshape(-1)!=0.]
        print("triplets after mask: ",triplets.shape)

    kgr_batch = 1024

    pos_label = np.ones((triplets.shape[0],1))
    pos_data = np.concatenate([triplets,pos_label],axis=-1)
    index = np.arange(len(pos_data))
    np.random.shuffle(index)
    pos_data = pos_data[index]

    pos_valid_num = int(len(pos_data) * 0.05)
    pos_train = pos_data[:-pos_valid_num]
    pos_valid = pos_data[-pos_valid_num:]

    # kgr_optimizer = torch.optim.AdamW(kgr_model.parameters(), lr=0.001)
    model.train()
    
    for epoch in tqdm(range(epochs)):
        torch.cuda.empty_cache()
        if epoch % 4 == 0:
            kgc_neg_data = _get_KGC_neg_data(triplets,neg_times=4) #book:4
            neg_label = np.zeros((kgc_neg_data.shape[0],1))
            neg_data = np.concatenate([kgc_neg_data,neg_label],axis=-1)
            
            neg_train = neg_data[:-pos_valid_num]
            neg_valid = neg_data[-pos_valid_num:]
            kgr_train_data = np.concatenate([pos_train, neg_train],axis=0)
            index = np.arange(len(kgr_train_data))
            np.random.shuffle(index)
            kgr_train_data = kgr_train_data[index]
            # kgr_valid_data = np.concatenate([pos_valid, neg_valid],axis=0)
            kgr_valid_data = pos_valid

            kgr_train_data = torch.LongTensor(kgr_train_data)
            kgr_valid_data = torch.LongTensor(kgr_valid_data)

            kgr_iter = math.ceil(len(kgr_train_data) / kgr_batch)
            kgr_val_iter = math.ceil(len(kgr_valid_data) / kgr_batch)

        batch_kg = dict()
        total_kg_loss = 0
        for i in range(kgr_iter):
            batch_kg["hr_pair"] = kgr_train_data[i * kgr_batch:(i + 1) * kgr_batch].to(device)

            kgr_batch_loss = model(batch_kg,mode="kgc")
            total_kg_loss += kgr_batch_loss.item()

            kgr_optimizer.zero_grad()
            kgr_batch_loss.backward()
            kgr_optimizer.step()
        print("epoch: ",epoch)
        print("KGR model total loss: ",total_kg_loss)

        print("evaluate model...")
        model.eval()
        pred_label = []
        true_label = []
        for i in range(kgr_val_iter):
            batch_kg["hr_pair"] = kgr_valid_data[i * kgr_batch:(i + 1) * kgr_batch].to(device)
            pre = model.gcn.kgc(batch_kg,eval=True)
            pre = (pre >= threshold).squeeze(-1).float()
            pre = pre.detach().cpu().numpy()
            label = batch_kg["hr_pair"][:,-1]
            label = label.cpu().numpy()
            pred_label.append(pre)
            true_label.append(label)
            
        pred_label = np.concatenate(pred_label,axis=0)
        print("pred_label: 1 shape:, ",pred_label[pred_label>0].shape)
        true_label = np.concatenate(true_label,axis=0)
        # pred_label_d = pred_label[pred_label>0]
        # true_label_d = true_label[pred_label>0]
        
        acc = accuracy_score(pred_label, true_label)
        print("evaluate acc: ",acc)
        

def _process_kg_attr(canditate_kg,tripltes,kg_mask=None):
    
    canditate_kg = np.unique(canditate_kg,axis=0)
    print("new kg num items: ",np.unique(canditate_kg[:,0]).shape)
    if kg_mask!=None:
        tripltes = tripltes[kg_mask.reshape(-1)!=0]
        
    # attr_triplets = tripltes[tripltes[:,0]>n_items-1]
    attr_set = set(np.unique(canditate_kg[:,-1]))
    out_attr_kg = []
    for tirp in tqdm(tripltes):
        if tirp[0] in attr_set:
            out_attr_kg.append(tirp)
    out_attr_kg = np.asarray(out_attr_kg)
    print("canditate_kg: ",canditate_kg.shape)
    print("out_attr_kg: ",out_attr_kg.shape)
    if out_attr_kg.shape[0] > 0:
        print("out_attr_kg: ",out_attr_kg.shape)
        all_candi_kg = np.concatenate([canditate_kg,out_attr_kg],axis=0)
    else:
        all_candi_kg = canditate_kg
    return np.unique(all_candi_kg,axis=0)
    


if __name__ == '__main__':
    """fix the random seed"""
    # seed = 1998
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
  
    """read args"""
    global args, device, train_user_set,kg_dict, item_lists_dict, ent_lists_dict
    args = parse_args()
    print(args)
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, ui_sparse_graph, all_sparse_graph, item_rel_mask, triplets, kg_dict = load_data(args)
    # item_pmi_dict = pkl.load(open("item_pair_pmi.pkl","rb"))
    item_pmi_dict = _cal_npmi(user_dict['train_user_set'])
    pkl.dump(item_pmi_dict,open(args.dataset + "item_pair_pmi.pkl","wb"))
    
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    train_user_set = user_dict['train_user_set']

    
    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    """define model"""
    model = Recommender(n_params, args, graph, ui_sparse_graph,item_rel_mask).to(device)
    # KGR_model = KGR(n_items, n_entities,n_relations,dim=256, p_norm=1,margin=1.).to(device)
    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = PCGrad(optimizer)
    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step,gamma=args.lr_dc)  
    kgr_optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)  
    
    cur_best = 0
    stopping_step = 0
    should_stop = False
    best_metric = {"recall":0, "ndcg":0, "precision":0, "hit_ratio":0}
    best_epoch  = {"recall":0, "ndcg":0, "precision":0, "hit_ratio":0}
    print("start training ...")
    
    iter = math.ceil(len(train_cf_pairs) / args.batch_size)
    cl_batch = 512
    cl_iter = math.ceil(n_items / cl_batch)
    item_embs = []
    KG_mask = []
    
    for epoch in range(100):
        torch.cuda.empty_cache()
        if epoch % 10 == 0 or epoch==0:
        # if epoch == 100 or epoch == 0:
            # shuffle training data
            index = np.arange(len(train_cf))
            np.random.shuffle(index)
            train_cf_pairs = train_cf_pairs[index]
            print("start prepare feed data...")
            all_feed_data = get_feed_data(train_cf_pairs, user_dict['train_user_set'])  # {'user': [n,], 'pos_item': [n,], 'neg_item': [n, n_sample]}
            all_feed_data['pos_index'] = torch.LongTensor(index)

        if epoch % 3 == 0:  #MIND: 8 others:3 
            candi_kg = _generate_candi_kg(item_pmi_dict,n_items,kg_dict,item_embeds=item_embs,pmi_threshold=0.6,cos_threshold=0.95) #yelp,MIND:0.7,others:0.5
            all_candi_kg = _process_kg_attr(candi_kg,triplets)
            if epoch == 0:
                kgr_epoch = 10    #MIND:6, others:8 book:10
            else:
                kgr_epoch = 1    #MIND:1, others:2
            train_kgr_model(model,kgr_optimizer, triplets,kg_mask=KG_mask, epochs=kgr_epoch)
            # fitered_kg = _generate_new_kgs(model,candi_kg,pre_rate=0.5)
            # fitered_kg = process_kg_attr(fitered_kg,triplets,kg_mask=KG_mask)
            # pkl.dump(fitered_kg,open(str(epoch)+"_filtered_kg","wb"))
            # fitered_kg = torch.LongTensor(fitered_kg).to(device)
            all_candi_kg = torch.LongTensor(all_candi_kg).to(device)
            model.gcn._update_knowledge(all_candi_kg)
            
        """training"""
        model.train()
        loss = 0
        train_s_t = time()

        for i in tqdm(range(iter)):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            batch = dict()
            batch['pos_index'] = all_feed_data['pos_index'][i * args.batch_size:(i + 1) * args.batch_size].to(device)
            batch['users'] = all_feed_data['users'][i*args.batch_size:(i+1)*args.batch_size].to(device)
            batch['pos_items'] = all_feed_data['pos_items'][i*args.batch_size:(i+1)*args.batch_size].to(device)

            batch_loss, batch_mmd_loss = model(batch)
            print("batch loss: ",batch_loss.item())
            # batch_loss.backward() 
            loss_list  = [batch_mmd_loss, batch_loss]
            optimizer.pc_backward(loss_list)
            optimizer.step()
            loss += batch_loss.item()
            print("epoch: ",epoch)

        # if epoch > 8:
        #     for param_group in optimizer.param_groups:
        #             param_group['lr'] = 0.0001

        train_e_t = time()
        # scheduler.step()                                         
        item_embs, KG_mask = model.generate(for_kgc=True)
        item_embs = item_embs.detach().cpu().numpy()
        KG_mask = KG_mask.detach().cpu().numpy()
        # if epoch > 4 :
        """testing"""
        model.eval()
        test_s_t = time()
        with torch.no_grad():
            ret = test(model, user_dict, n_params)
        test_e_t = time()
        for k in best_metric.keys():
            if ret[k][0] > best_metric[k]:
                best_metric[k] = ret[k][0]
                best_epoch[k] = epoch
        train_res = PrettyTable()
        train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
        train_res.add_row(
            [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss, ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
        )
        
        print(train_res)
        print("best metrics: ",best_metric)
        print("best epochs:  ",best_epoch)
        f = open('./result/{}_exp_cxks_kg_xr_v2.txt'.format(args.dataset), 'a+')
        f.write(str(best_metric)+ '\n')
        f.write(str(best_epoch)+ '\n')
        f.write(str(train_res) + '\n')
        f.write('\n')
        f.close()

        # *********************************************************
        cur_best, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best,
                                                                    stopping_step, expected_order='acc',
                                                                    flag_step=20)
        
        if should_stop:
            break
        """save model"""
        if ret['recall'][0] == cur_best and args.save:
            torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')
            
    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best))
