import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import  save_image
import torch.autograd
from torch.autograd import Variable
import os

#创建文件夹
if not os.path.exists('./img'):
    os.mkdir('./img')

def to_img(x):
    out = 0.5*(x+1)
    out = out.clamp(0,1)#Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 1, 28, 28)#view()函数作用是将一个多行的Tensor,拼接成一行
    return out

batch_size = 128 #一批128个

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#图形的处理过程
img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
    transforms.Normalize((0.1307,), (0.3081,))
])

#mnist dataset mnist数据集下载
mnist = datasets.MNIST(
    root='./data/mnist/', train=True, transform = img_transform, download = True
)

#data loader 数据载入(批次读取)
dataloader = torch.utils.data.DataLoader(
    dataset = mnist, batch_size = batch_size, shuffle = True
)


#################################   Discriminator   #########################################
#将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
# 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类。
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784,256),#输入特征数为784，输出为256
            nn.LeakyReLU(0.2),#进行非线性映射
            nn.Linear(256,256),#进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()#也是一个激活函数，二分类问题中，
            # sigmoid可以班实数映射到【0,1】，作为概率值，
            # 多分类用softmax函数
        )
    def forward(self, x):
        x = self.dis(x)
        return x


#################################   Generator   #########################################
# 输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,
# 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
# 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布
# 能够在-1～1之间。
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),  # 用线性变换将输入映射到256维
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 256),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 784),  # 线性变换
            nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间
        )

    def forward(self, x):
        x = self.gen(x)
        return x
