import csv
import random
import numpy as np
import torch
from DataAdapter import *
import torch.utils.data as Data
import visdom
from filter import filter

def read_data(batch_size,train_split,valid_split):

    signal_filt = []
    label = []

    train_data = r'..\data\train.csv'

    weight = [0,0,0,0]

    # 读取csv文件并统计权重
    with open(train_data,'r') as f:
        reader = csv.DictReader(f)
        for line in reader:
            signal_filt.append(filter([float(num) for num in line['heartbeat_signals'].split(',')]))
            label.append(int(float(line['label'])))
            weight[int(float(line['label']))] += 1

    # 使用pytorch的数据接口
    dataset = DataAdapter(signal_filt,label)
    # train_size = int(len(signal_filt) * train_split) - int(len(signal_filt) * valid_split)

    # 计算训练集与验证集大小
    train_size = int(len(signal_filt) * train_split)
    valid_size = len(signal_filt) - train_size
    # valid_size = int(len(signal_filt) * valid_split)
    # test_size = len(signal_filt) - train_size - valid_size
    # train_dataset,valid_dataset,test_dataset = Data.random_split(dataset,[train_size,valid_size,test_size])

    # 划分数据集
    train_dataset,valid_dataset = Data.random_split(dataset,[train_size,valid_size])

    train_loader = Data.DataLoader(train_dataset,batch_size = batch_size,shuffle = True,num_workers = 0)
    valid_loader = Data.DataLoader(valid_dataset,batch_size = batch_size,shuffle = True,num_workers = 0)
    # test_loader = Data.DataLoader(test_dataset,batch_size = batch_size,shuffle = False,num_workers = 0)

    print('Data Loading Finished')

    return train_loader,valid_loader,[len(label)/ii for ii in weight]

# 这个函数好像没用到，用于打乱数据的
def shuffle_data(train_loader,valid_loader,valid_split,batch_size):
    train_dataset = train_loader.dataset.dataset
    valid_dataset = valid_loader.dataset.dataset
    X = torch.cat((train_dataset.X,valid_dataset.X),0)
    Y = torch.cat((train_dataset.Y,valid_dataset.Y),0)
    dataset = DataAdapter(X,Y)
    train_dataset,valid_dataset = Data.random_split(dataset,[len(dataset) - int(len(dataset)*valid_split),int(len(dataset)*valid_split)])
    train_loader = Data.DataLoader(train_dataset,batch_size = batch_size,shuffle = True,num_workers = 0)
    valid_loader = Data.DataLoader(valid_dataset,batch_size = batch_size,shuffle = True,num_workers = 0)
    return train_loader,valid_loader