import numpy as np
import torch

def train_model(train_loader,model,criterion,optimizer,device):
    model.train()
    train_loss = []
    train_acc = []   

    for i,data in enumerate(train_loader,0):

        # inputs,labels = data[0].cuda(),data[1].cuda()
        inputs,labels = data[0].to(device),data[1].to(device) # 获取数据及标签

        outputs = model(inputs) # 前向传播得到结果

        _,pred = outputs.max(1) # 得到分类结果

        num_correct = (pred == labels).sum().item()
        acc = num_correct/len(labels) # 计算acc

        loss = criterion(outputs,labels) # 计算loss
        optimizer.zero_grad() # adam优化器梯度清零
        loss.backward() # 反向传播计算梯度
        optimizer.step() # 更新优化器参数
    
        train_loss.append(loss.item())
        train_acc.append(acc)

    return np.mean(train_loss),np.mean(train_acc)
