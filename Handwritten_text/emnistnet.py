import torch
import torch.nn as nn
import torch.nn.functional as F

 
class emnistnet(nn.Module):
        def __init__(self, printtoggle):
            super().__init__()
            
            self.print = printtoggle
            
            #1st convolution layer
            self.conv1 = nn.Conv2d(1,6,3, padding=1)
            self.bnorm1 = nn.BatchNorm2d(6)
            
            #2nd convoluton layer
            self.conv2 = nn.Conv2d(6,6,3, padding=1)
            self.bnorm2 = nn.BatchNorm2d(6)
            
            self.fc1 = nn.Linear(7*7*6,50)
            self.fc2 = nn.Linear(50,26)
            
        def forward(self,x):
            if self.print: print(f'Input: {list(x.shape)}')
                
            x = F.max_pool2d(self.conv1(x),2)
            x = F.leaky_relu(self.bnorm1(x))
            if self.print: print(f'first CPR block{list(x.shape)}')
                
            x = F.max_pool2d(self.conv2(x),2)
            x = F.leaky_relu(self.bnorm2(x))
            if self.print: print(f'second CPR block{list(x.shape)}')
                
            nUnits = x.shape.numel()/x.shape[0]
            x = x.view(-1,int(nUnits))
            if self.print: print(f'vectorized: {list(x.shape)}')
                
            x = F.leaky_relu(self.fc1(x))
            x = self.fc2(x)
            if self.print: print(f'final output: {list(x.shape)}')
                
            return x