import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class OccupancyRepair(nn.Module):
    def __init__(self, bottle_neck, image_width, image_height):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.code_size = bottle_neck
        
        self.enc_cnn_1 = nn.Conv2d(1, 10, kernel_size=5)
        self.enc_cnn_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.enc_linear_1 = nn.Linear(9680, 50)
        self.enc_linear_2 = nn.Linear(50, self.code_size)
        
        self.dec_linear_1 = nn.Linear(self.code_size, 160)
        self.dec_linear_2 = nn.Linear(160, image_width * image_height)
        
    def forward(self, x):
        encoding = self.encode(x)
        x = self.decode(encoding)
        return x, encoding
    
    def encode(self, x):
        x = self.enc_cnn_1(x)
        x = F.selu(F.max_pool2d(x, 2))
        x = self.enc_cnn_2(x)
        x = F.selu(F.max_pool2d(x, 2))
        x = x.view([x.size(0), -1])
        x = F.selu(self.enc_linear_1(x))
        x = self.enc_linear_2(x)
        return x
    
    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = F.sigmoid(self.dec_linear_2(out))
        out = out / out.max(1, keepdim=True)[0]
        out = out.view([code.size(0), 1, self.image_width, self.image_height])
        return out