import os
import re
from math import log
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader, random_split


def _generate_candi_kg(pmi_dict,n_items, kg_dict, item_embeds=[],pmi_threshold = 0.5, cos_threshold = 0.9):
    npmi_mat = np.zeros((n_items,n_items))    
    for item_pair in tqdm(pmi_dict.keys()):
        item_i, item_j = [int(x) for x in item_pair.split(",")]
        npmi = pmi_dict[item_pair][1]
        npmi_mat[item_i][item_j] = npmi_mat[item_i][item_j] + npmi
    npmi_mat = np.where(npmi_mat>=pmi_threshold,npmi_mat,0.)
    bool_mat = np.where(npmi_mat!=0,1,0.)
    print("pmi num pair: ",np.sum(bool_mat))
    
    # if item_embeds!=[]:
    #     embed_mat = cosine_similarity(item_embeds)
    #     embed_mat = np.where(embed_mat>=cos_threshold,embed_mat,0.)
    #     embed_bool_mat = np.where(embed_mat!=0,1.,0.)
    #     print("cos num pair: ",np.sum(embed_bool_mat))
    #     merge_mat = npmi_mat + embed_mat
    # else:
    merge_mat = npmi_mat
    
    sparse_adj = sp.coo_matrix(merge_mat)
    row_data = sparse_adj.row
    col_data = sparse_adj.col
    data_value = sparse_adj.data

    exp_row_data = np.expand_dims(row_data,axis=-1)
    exp_col_data = np.expand_dims(col_data,axis=-1)
    item_touples = np.concatenate([exp_row_data, exp_col_data],axis=-1)
    item_neb_dict = defaultdict(set)
    for touple in item_touples:
        item_1,item_2 = touple
        if item_1 not in item_neb_dict:
            item_neb_dict[item_1] = {item_2}
        else:
            item_neb_dict[item_1].add(item_2)
        
        if item_2 not in item_neb_dict:
            item_neb_dict[item_2] = {item_1}
        else:
            item_neb_dict[item_2].add(item_1)
            
    for k,v in item_neb_dict.items():
        item_neb_dict[k] = list(v)
    
    new_kg_dict = defaultdict(list)
    for k,v in tqdm(item_neb_dict.items(),desc="process item_touples..."):
        temp = []
        for item in v:
            kg_seq = kg_dict[item]
            for r_t in kg_seq:
                if r_t not in kg_dict[k]:
                    temp.append(r_t)
        new_kg_dict[k] = temp
    
    candi_kg = []
    for k, v in new_kg_dict.items():
        for r,t in v:
            candi_kg.append([k,r,t])
    candi_kg = np.array(candi_kg)
    candi_kg = np.unique(candi_kg,axis=0)    
    print("generate candi kg: ",candi_kg.shape[0])
    return candi_kg


def _cal_npmi(train_user_set):
    item_freq = {}
    for user, item_seq in train_user_set.items():
        appeared = set()
        for item in item_seq:
            if item in appeared:
                print(1)
                continue
            if item in item_freq:
                item_freq[item] += 1
            else:
                item_freq[item] = 1
            appeared.add(item)
            
    num_windows = len(train_user_set)
    item_pair_count = {}
    for user, item_seq in tqdm(train_user_set.items()):
        appeared = set()
        for i in range(1,len(item_seq)):
            for j in range(0,i):
                item_i = item_seq[i]
                item_j = item_seq[j]
            if item_i == item_j:
                continue
            item_pair_str = str(item_i) + "," +str(item_j)
            if item_pair_str in appeared:
                continue
            if item_pair_str in item_pair_count:
                item_pair_count[item_pair_str] += 1
            else:
                item_pair_count[item_pair_str] = 1
            appeared.add(item_pair_str)
            #reverse pair
            item_pair_str = str(item_j) + "," +str(item_i)
            if item_pair_str in appeared:
                continue
            if item_pair_str in item_pair_count:
                item_pair_count[item_pair_str] += 1
            else:
                item_pair_count[item_pair_str] = 1
            appeared.add(item_pair_str)

    tmp_max_npmi = 0
    tmp_min_npmi = 0
    tmp_max_pmi = 0
    tmp_min_pmi = 0    
    temp_item_i_max = None
    temp_item_j_max = None
    temp_i_j_max = None

    temp_item_i_min = None
    temp_item_j_min = None
    temp_i_j_min = None
    
    item_pair_pmi = defaultdict(list)
    print("calculating npmi...")
    for item_pair in tqdm(item_pair_count):
        item_i, item_j = [int(x) for x in item_pair.split(",")]
        pair_count = item_pair_count[item_pair]
        item_i_freq, item_j_freq = item_freq[item_i], item_freq[item_j]
        pmi = log(
        (1.0 * pair_count / num_windows)
        / (1.0 * item_i_freq * item_j_freq / (num_windows * num_windows))
        )
        npmi = (
            log(1.0 * item_i_freq * item_j_freq / (num_windows * num_windows))
            / log(1.0 * pair_count / num_windows)
            - 1
        )
        item_pair_pmi[item_pair] = [pmi, npmi, pair_count,item_i_freq,item_j_freq]

        if npmi > tmp_max_npmi:
            tmp_max_npmi = npmi
            temp_item_i_max = item_i_freq
            temp_item_j_max = item_j_freq
            temp_i_j_max = pair_count
        if npmi < tmp_min_npmi:
            tmp_min_npmi = npmi
            temp_item_i_min = item_i_freq
            temp_item_j_min = item_j_freq
            temp_i_j_min = pair_count

        if pmi > tmp_max_pmi:
            tmp_max_pmi = pmi
        if pmi < tmp_min_pmi:
            tmp_min_pmi = pmi
    print("max_pmi:", tmp_max_pmi, "min_pmi:", tmp_min_pmi)
    print("max_npmi:", tmp_max_npmi, "min_npmi:", tmp_min_npmi)
    print("temp_item_i_max: ",temp_item_i_max, "temp_item_j_max: ",temp_item_j_max, "temp_i_j_max: ",temp_i_j_max)
    print("temp_item_i_min: ",temp_item_i_min, "temp_item_j_min: ",temp_item_j_min, "temp_i_j_min: ",temp_i_j_min)
    return item_pair_pmi



def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

class MyDataset(Dataset):
    def __init__(self, train_cf_pairs, train_user_set, n_items, num_neg_sample):
        self.train_cf_pairs = train_cf_pairs
        self.train_user_set = train_user_set
        self.n_items = n_items
        self.num_neg_sample = num_neg_sample

    def __len__(self):
        return len(self.train_cf_pairs)

    def __getitem__(self, idx):
        ui = self.train_cf_pairs[idx]

        u = int(ui[0])
        each_negs = list()
        neg_item = np.random.randint(low=0, high=self.n_items, size=self.num_neg_sample)
        if len(set(neg_item) & set(self.train_user_set[u])) == 0:
            each_negs += list(neg_item)
        else:
            neg_item = list(set(neg_item) - set(self.train_user_set[u]))
            each_negs += neg_item
            while len(each_negs) < self.num_neg_sample:
                n1 = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if n1 not in self.train_user_set[u]:
                    each_negs += [n1]


        return [ui[0], ui[1], each_negs]