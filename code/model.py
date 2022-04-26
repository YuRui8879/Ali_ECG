import torch
import torch.nn as nn
import torch.nn.functional as F

# 模型结构文件
class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 1,out_channels = 32,kernel_size = 11,stride = 1,padding = 5) # 一维卷积，卷积核数量是32，卷积核大小是11，步长是1，padding是5
        self.conv2 = nn.Conv1d(32,64,11,1,5) # 一维卷积，卷积核数量是64，卷积核大小是11，步长1，padding5
        self.conv3 = nn.Conv1d(64,128,3,1,1)
        self.conv4 = nn.Conv1d(128,256,3,1,1)
        self.bn1 = nn.BatchNorm1d(1) # 批标准化
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.maxpool = nn.MaxPool1d(4) # 最大池化
        self.relu = nn.ReLU() # ReLU激活函数
        self.dropout = nn.Dropout(0.1) 
        self.linear_unit = nn.Sequential(
            nn.Linear(3072,1024), # 全连接层
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,4),
            nn.Softmax(dim=1) # softmax激活函数
        )

    def forward(self,x):
        x = x.view(x.size(0),1,x.size(1))
        x = self.bn1(x)
        x = self.relu(self.conv1(x))
        x = self.bn2(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.bn3(x)
        x = self.relu(self.conv3(x))
        x = self.bn4(x)
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0),-1)
        x = self.linear_unit(x)
        return x

class CNN2(nn.Module):

    def __init__(self):
        super(CNN2,self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 1,out_channels = 32,kernel_size = 11,stride = 1,padding = 5)
        self.conv2 = nn.Conv1d(32,64,11,1,5)
        self.conv3 = nn.Conv1d(64,128,3,1,1)
        self.conv4 = nn.Conv1d(128,256,3,1,1)
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear_unit = nn.Sequential(
            nn.Linear(3072,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,4),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = x.view(x.size(0),1,x.size(1))
        x = self.bn1(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.bn2(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0),-1)
        x = self.linear_unit(x)
        return x

class CNN3(nn.Module):

    def __init__(self):
        super(CNN3,self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 1,out_channels = 32,kernel_size = 11,stride = 1,padding = 5)
        self.conv2 = nn.Conv1d(32,64,11,1,5)
        self.conv3 = nn.Conv1d(64,128,3,1,1)
        self.conv4 = nn.Conv1d(128,256,3,1,1)
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.maxpool = nn.MaxPool1d(4)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear_unit = nn.Sequential(
            nn.Linear(3072,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,128),
            nn.LeakyReLU(),
            nn.Linear(128,4),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = x.view(x.size(0),1,x.size(1))
        x = self.bn1(x)
        x = self.relu(self.conv1(x))
        x = self.bn2(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.bn3(x)
        x = self.relu(self.conv3(x))
        x = self.bn4(x)
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0),-1)
        x = self.linear_unit(x)
        return x

# x = torch.rand(5,205)
# model = CNN()
# y = model.forward(x)
# print(y[0][0])
