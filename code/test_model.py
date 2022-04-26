import numpy as np
import torch

def test_model(test_loader,criterion,model,device):
    model.eval()
    test_loss = []
    test_acc = []   

    for i,data in enumerate(test_loader,0):

        # inputs,labels = data[0].cuda(),data[1].cuda()
        inputs,labels = data[0].to(device),data[1].to(device) # 获得数据及标签

        outputs = model(inputs) # 前向传播计算结果
        loss = criterion(outputs,labels) # 计算loss

        _,pred = outputs.max(1) # 得到分类结果

        num_correct = (pred == labels).sum().item()
        acc = num_correct/len(labels) # 计算acc
        
        test_loss.append(loss.item())
        test_acc.append(acc)

    return np.mean(test_loss),np.mean(test_acc)
