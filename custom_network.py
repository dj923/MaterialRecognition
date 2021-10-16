import torch
import numpy as np
import cv2
import copy
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Normalize(nn.Module):
    
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
    """
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, eps=1e-8)


class oneStream_multiView_Net(nn.Module):
    def __init__(self,numClass):
        super(oneStream_multiView_Net, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512,128)

        self.lstm_1 = nn.LSTM(input_size=128, hidden_size=128,num_layers=1,batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=128, hidden_size=75,num_layers=1,batch_first=True)
        self.fc1 = nn.Linear(75,numClass)

    def forward(self, x):
        b_size, numView, C, H, W = x.size()
        x = x.view(b_size*numView, C, H, W) 
        x = self.model(x)
        x = x.view(b_size, numView, -1) # reshape input for LSTM

        x,(_,_) = self.lstm_1(x)
        x,(_,_) = self.lstm_2(x)

        x = self.fc1(x[:,-1,:])
        return x


class twoStream_multiView_Net(nn.Module):
    def __init__(self,numClass):
        super(twoStream_multiView_Net, self).__init__()
        self.model1 = models.resnet18(pretrained=True)
        self.model2 = models.resnet18(pretrained=True)

        self.model1.fc = nn.Linear(512,128)
        self.model2.fc = nn.Linear(512,64)

        self.lstm_1 = nn.LSTM(input_size=192, hidden_size=128,num_layers=1,batch_first=True)
        self.fc1 = nn.Linear(128,numClass)

        self.L2norm = Normalize()
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        cuda = torch.device('cuda')
        b_size, numView, C, H, W = x.size()
        #===========Patch Sorting===========#
        x_1 = copy.copy(x)
        m = torch.zeros((b_size,numView))
        c = torch.zeros((b_size,numView),dtype =torch.int8)

        for i in range(b_size):
            for j in range(numView):
                m[i,j] = torch.mean(x[i,j,:,:,:])
        for i in range(b_size):
            for j in range(numView):
                r=0
                for k in range(numView):
                    if m[i,j] < m[i,k] : r+=1
                c[i,j] = r
        for i in range(b_size):
            for j in range(numView):
                x[i,c[i,j]] = x_1[i,j]
        #==================================#

        x = x.reshape(b_size*numView, C, H, W) # reshape input for CNN: rnnBatch*25 = cnnBatch

        #=========Making reflectance Patch=========#
        x_r = copy.copy(x).reshape(b_size,numView,C,H,W) 
        x_avg = torch.zeros((b_size,C,H,W))
        for i in range(b_size):
            x_avg[i] = torch.mean(x_r[i],dim=0)
        x_avg = x_avg.reshape(b_size,C,H,W)

        for a in range(numView):    
            x_r[:,a] = torch.from_numpy(cv2.absdiff(np.array(x_avg[:].cpu()),np.array(x_r[:,a].cpu())))
        x_r = x_r.reshape(b_size*numView, C, H, W)
        #============================+++============#

        x = self.model1(x)
        x_r = self.model2(x_r)
        x = x.view(b_size, numView, -1) 
        x_r = x_r.view(b_size, numView, -1) 

        x = torch.cat([x,x_r],dim=2)
        del x_r
        x = x.view(b_size, numView, -1) # reshape input for LSTM: cnnBatch/25 = rnnBatch

        x,(_,_) = self.lstm_1(x)
        x = self.fc1(x[:,-1,:])
        
        return x


