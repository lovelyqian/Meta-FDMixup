
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms



class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
  
        # classifier
        self.domain_fc = nn.Linear(64, 2)
      
    def forward(self, x):
        out = self.domain_fc(x)
        return out




if __name__=='__main__':
    print('hello')
    x = torch.randn(16, 64)
    x = Variable(x).cuda()

    my_model = DomainClassifier()
    my_model = my_model.train().cuda()
    out = my_model(x)
    print('out:', out.size())
 
