import csv
import os
import pandas as pd
import torch
from filter import filter
from model import CNN
from DataAdapter import DataAdapter
import torch.utils.data as Data
import numpy as np

def predict_ali_testset(batch_size,model,device):
    ipath = r'..\data'
    opath = r'..\output'
    signal = []

    # 读取数据集B
    with open(os.path.join(ipath,'testB.csv'),'r') as f:
        reader = csv.DictReader(f)
        for line in reader:
            signal.append(filter([float(num) for num in line['heartbeat_signals'].split(',')]))
    
    # 加载测试集数据
    test_set = DataAdapter(signal,[0 for i in range(len(signal))])
    test_loader = Data.DataLoader(test_set,batch_size = batch_size,shuffle = False, num_workers = 0)

    res = [[],[],[],[]]

    with torch.no_grad():
        for i,data in enumerate(test_loader):
            # inputs,_ = data[0].cuda(),data[1].cuda()
            inputs,_ = data[0].to(device),data[1].to(device)

            temp_output = []
            temp_pred = []
            # 每一个模型均输出一个结果
            for m in model:
                outputs = m(inputs)
                temp_output.append(outputs)
                _,pred = outputs.max(1)
                temp_pred.append(np.array(pred.cpu()))
            #使用投票法得到预测结果
            temp_pred = np.transpose(np.array(temp_pred))
            for ii in range(len(temp_pred)):
                line_pred = temp_pred[ii,:]
                max_num = list(np.bincount(line_pred)).index(np.max(np.bincount(line_pred)))
                max_output = temp_output[0][ii]
                for l in range(len(line_pred)):
                    if max_num == line_pred[l]:
                        if temp_output[l][ii][max_num].cpu().numpy() > max_output[max_num].cpu().numpy():
                            max_output = temp_output[l][ii]
                outputs[ii] = torch.cuda.FloatTensor(max_output)
            
            
            res[0].extend(outputs[:,0].cpu().numpy())
            res[1].extend(outputs[:,1].cpu().numpy())
            res[2].extend(outputs[:,2].cpu().numpy())
            res[3].extend(outputs[:,3].cpu().numpy())

    # 写入预测概率
    result = pd.read_csv(os.path.join(ipath,'sample_submit.csv'))
    result['label_0'] = res[0]
    result['label_1'] = res[1]
    result['label_2'] = res[2]
    result['label_3'] = res[3]
    result['id'] = [i for i in range(120000,140000)]
    result.to_csv(os.path.join(opath,'sample_submit.csv'),index = False)

    print('预测文件写入完成')

# device = torch.device('cuda:1')
# batch_size = 128
# predict_ali_testset(batch_size,device)