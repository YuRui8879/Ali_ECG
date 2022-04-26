from model import CNN,CNN2,CNN3
from test_model_for_ali import test_model_for_ali
from predict_ali_testset import predict_ali_testset
import torch

# 训练设备，根据自己的电脑进行设置
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 超参数
batch_size = 128
folds = 15

# 初始化模型
model = [CNN() for i in range(5)]
model.extend([CNN2() for i in range(5)])
model.extend([CNN3() for i in range(5)])

# 加载模型
for fold in range(folds):
    if fold < 5:
        model[fold].load_state_dict(torch.load(r'..\output\159.98\model' + 'best_model_' + str(fold) + '.pt'))
    elif fold < 10:
        model[fold].load_state_dict(torch.load(r'..\output\170\model' + 'best_model_' + str(fold-5) + '.pt'))
    else:
        model[fold].load_state_dict(torch.load(r'..\output\173\model' + 'best_model_' + str(fold-10) + '.pt'))
    model[fold].eval()
    model[fold].to(device)

# 重新保存一下，好像是之前保存的模型结构有问题，应该跑一次就行
for i in range(15):
    if i < 5:
        model_path = r'..\output\159\best_model_' + str(i) + '.pt'
    elif i <10:
        model_path = r'..\output\170\best_model_' + str(i-5) + '.pt'
    else:
        model_path = r'..\output\173\best_model_' + str(i-10) + '.pt'
    torch.save(model[i].state_dict(), model_path, _use_new_zipfile_serialization=False)

# 输出预测文件
predict_ali_testset(batch_size,model,device)