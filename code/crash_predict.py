from model import CNN,CNN2
from test_model_for_ali import test_model_for_ali
from predict_ali_testset import predict_ali_testset
import torch

# 训练设备设置，根据自己的设备进行设置
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 超参数设置
batch_size = 128
folds = 10

# 初始化模型
model = [CNN() for i in range(5)]
model.extend([CNN2() for i in range(5)])

# 加载模型
for fold in range(folds):r
    if fold < 5:
        model[fold].load_state_dict(torch.load(r'..\output\159.98\model' + 'best_model_' + str(fold) + '.pt'))
    else:
        model[fold].load_state_dict(torch.load(r'..\output\170\model' + 'best_model_' + str(fold-5) + '.pt'))
    model[fold].eval()
    model[fold].to(device)
r
# 输出预测文件
predict_ali_testset(batch_size,model,device)