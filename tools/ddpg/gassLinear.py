import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
from torch import optim
import torch.utils.data as Data
 
# def gaussian(in_features,out_features,signal):
#     SNR = 5
#      	#产生N(0,1)噪声数据
#     if out_features == 0:
#         noise = np.random.randn(in_features)
#     else:
#         noise = np.random.randn(in_features,out_features)
#     noise = noise-np.mean(noise) 								#均值为0
#     signal_power = np.linalg.norm( signal )**2 / signal.size	#此处是信号的std**2
#     noise_variance = signal_power/np.power(10,(SNR/10))         #此处是噪声的std**2
#     noise = (np.sqrt(noise_variance) / np.std(noise) )*noise    ##此处是噪声的std**2
#     return noise

def gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins + noise
    return ins

# 定义DisMult层
class GassLinear(nn.Module):
    def __init__(self,in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
		# nn.Module子类的函数必须在构造函数中执行父类的构造函数
		# 下式等价于nn.Module.__init__(self)
        super(GassLinear, self).__init__()
		# 隐特征维度
        self.u_W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_W = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.u_B = nn.Parameter(torch.randn(out_features))
        self.sigma_B = nn.Parameter(torch.randn(out_features))
		# 关系特定的方阵
		# self.weights = nn.Parameter(torch.Tensor(emb_size, emb_size), requires_grad=requires_grad)
		
		# 初始化参数
        self.reset_parameters()
 
	# 初始化参数
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.u_W.size(0))
        self.u_W.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.sigma_W.size(0))
        self.sigma_W.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.u_B.size(0))
        self.u_B.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.sigma_B.size(0))
        self.sigma_B.data.uniform_(-stdv, stdv)
        
    def reset_sigma(self):
#         stdv = 1. / math.sqrt(self.u_W.size(0))
#         self.u_W.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.sigma_W.size(0))
        self.sigma_W.data.uniform_(-stdv, stdv)
#         stdv = 1. / math.sqrt(self.u_B.size(0))
#         self.u_B.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.sigma_B.size(0))
        self.sigma_B.data.uniform_(-stdv, stdv)
 
	# 前向传播函数
    def forward(self, input1):
		# 前向传播的逻辑
#         print("input",input1.shape)
#         temp =self.u_W + self.sigma_W*torch.tensor(gaussian(self.out_features, self.in_features)).cuda()
#         print("temp1",temp.shape)
#         temp2 = self.u_B + self.sigma_B*torch.tensor(gaussian(self.out_features, 0)).cuda()
#         print("temp2",temp2.shape)
#         input1 = input1@(self.u_W + self.sigma_W*torch.tensor(gaussian(self.out_features, self.in_features,self.sigma_W.cpu().detach().numpy())).float().cuda()).t() + self.u_B + self.sigma_B*torch.tensor(gaussian(self.out_features, 0,self.sigma_W.cpu().detach().numpy())).float().cuda()
        input1 = input1@(self.u_W + self.sigma_W*gaussian(self.sigma_W,1,0,1)).t() + self.u_B + self.sigma_B*gaussian(self.sigma_B,1,0,1)
        return input1