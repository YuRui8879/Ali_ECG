import torch.nn as nn
import torch.optim as optim 
from train_model import *
from test_model import *
from read_data import read_data,shuffle_data
from test_model_for_ali import test_model_for_ali
import time
import visdom
from model import CNN
from predict_ali_testset import predict_ali_testset
from Log import Log
from EarlyStopping import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR

log_path = r'..\log'  # 日志文件路径
log = Log('test4',log_path,'w') # 日志类实例化
#训练可视化
vis = visdom.Visdom(port = 8007) #python -m visdom.server -p 8007  建立visdom本地虚拟服务器的代码 cmd下运行

#设置训练显卡
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # 我的电脑里有2张显卡，这里选择第二张，如果只有1张显卡，改为cuda:0即可
log.log(device) # 输出使用的设备

# visdom初始化，用于绘图
def visdom_init(): 
    loss_curve = vis.line(
        X = np.array( [0] ),
        Y = np.array( [[0,0]] ),
        opts = dict(
            xlabel='epoch',
            ylabel='train_loss',
            legend=['train_loss','valid_loss'])
    )
    acc_curve = vis.line(
        X = np.array( [0] ),
        Y = np.array( [[0,0]] ),
        opts = dict(
            xlabel='epoch',
            ylabel='acc',
            legend=['train_acc','valid_acc'])
    )
    return loss_curve,acc_curve

# visdom增加数据点函数，用于绘图
def visdom_change(epoch,train_loss,valid_loss,train_acc,valid_acc,loss_curve,acc_curve,log):
    try:                                         
        vis.line(
                X = [epoch],
                Y = [[train_loss,valid_loss]],
                win = loss_curve,
                update='append'
                )    
    except:
        log.log('visdom error......')

    try:                                       
        vis.line(
                X = [epoch],
                Y = [[train_acc,valid_acc]],
                win = acc_curve,
                update='append'
                )      
    except:
        log.log('visdom error......')

#超参数
batch_size = 128 # 批大小
folds = 1 # 折数，用于训练集成模型，设置为1则代表不使用集成模型
patience = 50 # 提前停止的参数，这里表示50代无提升则停止
# model = [nn.DataParallel(CNN()).cuda() for i in range(folds)]
model = [CNN().to(device) for i in range(folds)] # 初始化模型
criterion = nn.CrossEntropyLoss() # 使用交叉熵作为损失函数
train_loader,valid_loader,weight = read_data(batch_size,0.8,0.1) # 读取数据

log.log(model[0]) # 打印模型结构

for fold in range(folds):
    early_stopping = EarlyStopping(log,patience, verbose=False) # 提前停止类实例化
    train_loader,valid_loader = shuffle_data(train_loader,valid_loader,0.1,batch_size) # 打乱数据
    #weight = torch.FloatTensor(weight).to(device)
    loss_curve,acc_curve = visdom_init() # 初始化绘图

    optimizer = optim.Adam(model[fold].parameters(), lr=0.0001) # 使用Adam作为梯度下降算法
    clr = CosineAnnealingLR(optimizer,T_max = 32) # 使用余弦退火算法作为动态学习率
    best_loss = 3 # 记录最好的loss

    for epoch in range(1,201):
        time_all=0
        start_time = time.time()
        train_loss,train_acc = train_model(train_loader, model[fold], criterion, optimizer,device) # 训练模型
        clr.step()
        time_all = time.time()-start_time
        valid_loss,valid_acc = test_model(valid_loader,criterion,model[fold],device) # 验证模型
        # 打印参数
        log.log('- Epoch: %d - Train_loss: %.5f - Train_acc: %.5f - Val_loss: %.5f - Val_acc: %5f - T_Time: %.3f' %(epoch,train_loss,train_acc,valid_loss,valid_acc,time_all))
        log.log('当前学习率：%f' %optimizer.state_dict()['param_groups'][0]['lr'])
        
        # 绘图更新
        visdom_change(epoch,train_loss,valid_loss,train_acc,valid_acc,loss_curve,acc_curve,log)
            
        # 如果当前模型更优，则保存当前模型
        if valid_loss < best_loss:
            best_loss = valid_loss
            log.log('Find better model in Epoch {0}, saving model.'.format(epoch))
            torch.save(model[fold].state_dict(), log.get_model_path() + 'best_model_' + str(fold) + '.pt')

        # 计算提前停止条件
        early_stopping(valid_loss, model[fold])
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            log.log("Early stopping")
            # 结束模型训练
            break

    log.log('Fold %d Training Finished' %(fold+1))
    torch.cuda.empty_cache()

# 训练结束
log.log('Training Finished')

# 初始化模型
model = [CNN() for i in range(folds)]
# 加载最优模型
for fold in range(folds):
    model[fold].load_state_dict(torch.load(log.get_model_path() + 'best_model_' + str(fold) + '.pt'))
    model[fold].eval()
    model[fold].to(device)

# 使用阿里的计算方法计算指标
test_sum,test_acc = test_model_for_ali(test_loader,model,device)
print(test_sum)
log.log('test_acc %.4f' %test_acc)
log.close_file()

# 输出预测文件
predict_ali_testset(batch_size,model,device)