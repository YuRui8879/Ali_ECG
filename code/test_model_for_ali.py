import numpy as np
import torch

# 该函数与predict_ali_testset.py类似，也是用投票法得到结果，只是没有输出预测文件，而是计算阿里的评价指标以及acc
def test_model_for_ali(test_loader,model,device):
    
    test_acc = []  
    sums = 0

    with torch.no_grad():
        for i,data in enumerate(test_loader,0):

            # inputs,labels = data[0].cuda(),data[1].cuda()
            inputs,labels = data[0].to(device),data[1].to(device)
            
            temp_output = []
            temp_pred = []
            for m in model:
                outputs = m(inputs)
                temp_output.append(outputs)
                _,pred = outputs.max(1)
                temp_pred.append(np.array(pred.cpu()))
            
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

            sums += torch.sum(torch.sum(torch.abs(outputs - torch.nn.functional.one_hot(labels,4)))).cpu()

            num_correct = (pred == labels).sum().item()
            acc = num_correct/len(labels)
            
            test_acc.append(acc)

    return sums,np.mean(test_acc)