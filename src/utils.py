
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
# read data from files
# @return (numpy array) data, label, len
def read_data(file_path):

    if not os.path.exists(file_path):
        return None, None, 0

    with open(file_path, 'rb') as fr:
        data_set = pickle.load(fr)
        size = len(data_set[0])
        list_data = []
        list_label = []
        # illegal data
        if not len(data_set[0]) == len(data_set[1]):
            return None, None, 0

        #data = data_set[0][:size] / 255.

        data = torch.unsqueeze(data_set[0], dim=1).type(torch.FloatTensor)[:size]
        label = data_set[1][:size]

        data = np.asarray(data)
        label = np.asarray(label)
        return data, label, size

def read_data_label(data_path, label_path):

    if not os.path.exists(data_path):
        return None, None, 0

    with open(data_path, 'rb') as fr:
        test_data = pickle.load(fr)
        size = len(test_data)
    with open(label_path, 'rb') as fr:
        test_label = pickle.load(fr)
    return test_data, test_label, size

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0)

def cal_lp_norm(tensor,p,dim_count):
    tmp = tensor
    for i in range(1,dim_count):
        tmp = torch.norm(tmp,p=p,dim=i,keepdim=True)
    return tmp

# --0-30-60-90-120--, lr/10
# 40 clean + 30 anp + 30 anp + 20 clean
# return : epsilon, alpha, pro_num
def set_anp(epoch):
    if epoch < 40:
        return 0,0,1
    elif epoch < 60:
        return 0.3,0.7,3
    elif epoch < 80:
        return 1.0,1.0,3
    elif epoch < 100:
        return 0.3,0.7,3
    elif epoch < 120:
        return 0,0,1

def save_loss_acc(path,train_losses,train_acc,test_acc,train_step,test_step):
    with open(path + 'train_losses.p','wb') as f:
        pickle.dump(train_losses, f, pickle.HIGHEST_PROTOCOL)
    with open(path + 'train_acc.p','wb') as f:
        pickle.dump(train_acc, f, pickle.HIGHEST_PROTOCOL)
    with open(path + 'test_acc.p','wb') as f:
        pickle.dump(test_acc, f, pickle.HIGHEST_PROTOCOL)
    with open(path + 'train_step.p','wb') as f:
        pickle.dump(train_step, f, pickle.HIGHEST_PROTOCOL)
    with open(path + 'test_step.p','wb') as f:
        pickle.dump(test_step, f, pickle.HIGHEST_PROTOCOL)








