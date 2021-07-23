
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms


class Disentangle(nn.Module):
    def __init__(self):
        super(Disentangle, self).__init__()

        #encoder
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256, momentum=0.1)

        self.fc21a = nn.Linear(256, 64)
        self.fc22a = nn.Linear(256, 64)
        self.fc21b = nn.Linear(256, 64)
        self.fc22b = nn.Linear(256, 64)



    def encode(self, x):
        h1 = F.relu(self.bn1(self.fc1(x)))
       
        # a encoder: domain irrelevant
        a_mean, a_logvar = self.fc21a(h1), self.fc22a(h1)
        
        # b encoder: domain specific
        b_mean, b_logvar = self.fc21b(h1), self.fc22b(h1)
 
        return a_mean, a_logvar, b_mean, b_logvar


    def reparametrize(self, mu,logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


    def forward(self, x):
        a_mu, a_logvar, b_mu, b_logvar = self.encode(x)
        a_fea = self.reparametrize(a_mu, a_logvar)             # domain-irrelevant  (H1)
        b_fea = self.reparametrize(b_mu, b_logvar)             # domain-specific    (H2)
        return a_fea, b_fea




if __name__=='__main__':
    print('hello')
    x = torch.randn(16, 512)
    x = Variable(x).cuda()

    VAE_model = VAE()
    VAE_model = VAE_model.train().cuda()
    x_recon, mu, logvar, b_mu, b_logvar = VAE_model(x)
    print('x_recon:', x_recon.size(), 'mu:', mu.size(), 'logvar:', logvar.size())
 
    loss_vae = VAE_loss(x_recon, x, mu, logvar, b_mu, b_logvar)
    print('loss_varL', loss_vae)
