import torch.nn as nn
from tensorboardX import SummaryWriter

from methods import backbone
from methods.disentangle_module import Disentangle
from methods.domain_classifier import DomainClassifier



# --- conventional supervised training ---
class BaselineTrain(nn.Module):
  def __init__(self, model_func, num_class, tf_path=None, loss_type = 'softmax'):
    super(BaselineTrain, self).__init__()

    # feature encoder
    self.feature    = model_func()

    # disentnagle module
    self.disentangle_model = Disentangle()

    self.feature.final_feat_dim = 64

    if loss_type == 'softmax':
      self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
      self.classifier.bias.data.fill_(0)

    elif loss_type == 'dist':
      self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)

    self.loss_type = loss_type
    self.loss_fn = nn.CrossEntropyLoss()

    self.num_class = num_class
    self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None


  def forward_loss(self, x, y):
    # forward feature extractor
    x = x.cuda()
    x_fea  = self.feature.forward(x)
    
    # forward disentangle module
    a_code, b_code = self.disentangle_model(x_fea)
    
    # forward the classifier using the domain-irrelevant a code
    CLS_scores = self.classifier.forward(a_code) 

    # calculate loss
    y = y.cuda()
    loss_CLS = self.loss_fn(CLS_scores, y)

    return loss_CLS


  def train_loop(self, epoch, train_loader, optimizer, total_it):
    print_freq = len(train_loader) // 10
    avg_loss=0

    for i, (x,y) in enumerate(train_loader):
      optimizer.zero_grad()
      loss = self.forward_loss(x, y)
      loss.backward()
      optimizer.step()

      avg_loss = avg_loss+loss.item()#data[0]

      if (i + 1) % print_freq==0:
        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader), avg_loss/float(i+1)  ))
      if (total_it + 1) % 10 == 0:
        self.tf_writer.add_scalar('loss', loss.item(), total_it + 1)
      total_it += 1
    return total_it

  def test_loop(self, val_loader):
    return -1             #no validation, just save model during iteration

