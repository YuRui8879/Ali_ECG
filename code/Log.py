import os
import time
import matplotlib.pyplot as plt

# 日志类，可以直接使用，如果不想使用，主函数的全部的log改为print即可
class Log:

    def __init__(self,name,record_path,mode):
        record_path = record_path + name
        if not os.path.exists(record_path):
            os.makedirs(record_path)
        self.record_path = record_path + '\\'

        log_path = self.record_path + 'log'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.log_path = log_path + '\\'
        fullpath = self.log_path + name + '_log.txt'

        fig_path = self.record_path + 'fig'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        self.fig_path = fig_path + '\\'

        mid_path = self.record_path + 'mid'
        if not os.path.exists(mid_path):
            os.makedirs(mid_path)
        self.mid_path = mid_path + '\\'

        model_path = self.record_path + 'model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model_path = model_path + '\\'
        
        if mode == 'w':
            if (name + '.txt') in os.listdir(self.log_path):
                raise Exception('存在同名文件，程序终止')
            try:
                self.f = open(fullpath,'w+')
                print('以写模式打开日志文件成功')
                self.f.writelines(['[ ',self.get_time(),' ]','[ System] ','打开日志成功'])
                self.f.write('\n')
                self.f.writelines(['[ ',self.get_time(),' ]','[ System] ','写模式'])
                self.f.write('\n')
            except(IOError):
                print('日志文件打开失败')
        elif mode == 'a':
            try:
                self.f = open(fullpath,'a+')
                print('以追加模式打开日志文件成功')
                self.f.writelines(['[ ',self.get_time(),' ]','[ System] ','打开日志成功'])
                self.f.write('\n')
                self.f.writelines(['[ ',self.get_time(),' ]','[ System] ','追加模式'])
                self.f.write('\n')
            except(IOError):
                print('日志文件打开失败')
        else:
            print('未知模式，打开文件失败')

# public接口
# 打印消息，同时输出到日志文件

    def log(self,message,level = 'info'):
        if level == 'onscreen':
            print(message)
        elif level == 'ontext':
            print('[ ' + self.get_time() + ' ]' + '[  Info ]',end=' ',file=self.f)
            print(message,file=self.f)
        elif level == 'info':
            print(message)
            print('[ ' + self.get_time() + ' ]' + '[  Info ]',end=' ',file=self.f)
            print(message,file=self.f)
        elif level == 'warning':
            print(message)
            print('[ ' + self.get_time() + ' ]' + '[Warning]',end=' ',file=self.f)
            print(message,file=self.f)
        elif level == 'error':
            print(message)
            print('[ ' + self.get_time() + ' ]' + '[ Error ]',end=' ',file=self.f)
            print(message,file=self.f)
        elif level == 'none':
            pass
        else:
            print('Unknow level, failed to log')

# 关闭文件

    def close_file(self):
        try:
            self.f.writelines(['[ ',self.get_time(),' ]','[ System] ','关闭日志文件','\n'])
            self.f.write('-\n')
            self.f.write('-\n')
            self.f.close()
            print('日志文件正常关闭')
        except(IOError):
            print('日志文件关闭异常')

# 绘图

    def draw(self,name,x_label,is_show = True,**args):
        plt.figure()
        plt.xlabel(x_label)
        plt.title(name)
        plt.grid(True)
        for k,v in args.items():
            plt.plot(range(len(list(v))),list(v),label = k)
        plt.legend(loc = 'upper right')
        plt.savefig(self.fig_path + name + '.png')
        if is_show:
            plt.show()

    def get_seq_save_path(self):
        return self.mid_path

    def get_model_path(self):
        return self.model_path

    def get_fig_path(self):
        return self.fig_path
        
# private函数

    def get_time(self):
        timestamp = time.time()
        locaoltime = time.localtime(timestamp)
        return time.strftime('%Y-%m-%d %H:%M:%S',locaoltime)

def log_test():
    log = Log('test','./','a')
    corr = 4.555264
    log.log('corr: %s' %(str(round(corr,4))))
    log.close_file()

def draw_test():
    log = Log('test','./','a')
    log.draw('./test','epoch',train = [1,2,3,4,5,6,7,8,9,10],test = [10,9,8,7,6,5,4,3,2,1])
    log.close_file()

if __name__ == "__main__":
    log_test()
    draw_test()
