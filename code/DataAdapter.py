import torch.utils.data as Data
import torch

# pytorch数据接口
class DataAdapter(Data.Dataset):

    def __init__(self,X,Y):
        super(DataAdapter,self).__init__()
        self.X = torch.FloatTensor(X)
        self.Y = torch.LongTensor(Y)

    def __getitem__(self,index):
        return self.X[index,:],self.Y[index]

    def __len__(self):
        return len(self.X)


