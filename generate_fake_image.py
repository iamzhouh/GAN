import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import  save_image
import torch.autograd
from torch.autograd import Variable
import os

from model import generator

def to_img(x):
    out = 0.5*(x+1)
    out = out.clamp(0,1)#Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 1, 28, 28)#view()函数作用是将一个多行的Tensor,拼接成一行
    return out

G = generator()

G.load_state_dict(torch.load('generator.pth', map_location='cpu'))
G.eval()

z = Variable(torch.randn(10, 100))

fake_img = G(z)

fake_images = to_img(fake_img.cpu().data)
save_image(fake_images, './fake_images.png')