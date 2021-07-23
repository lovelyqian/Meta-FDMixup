import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from mixup import mixup_data
from methods import backbone
from methods.gnn import GNN_nl
from methods.disentangle_module import Disentangle
from methods.domain_classifier import DomainClassifier
from methods.meta_template import MetaTemplate





class MetaFDMixup(MetaTemplate):
  maml=False
  def __init__(self, model_func,  n_way, n_support, tf_path=None):
    super(MetaFDMixup, self).__init__(model_func, n_way, n_support, tf_path=tf_path)

    # loss function
    self.loss_fn = nn.CrossEntropyLoss()
    self.loss_KLD = nn.KLDivLoss()

    # disentangle  model
    self.disentangle_model = Disentangle()

    # DomainClassifer model
    self.domain_model = DomainClassifier()

    # metric function
    self.feat_dim = 64
    self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.BatchNorm1d(128, track_running_stats=False)) if not self.maml else nn.Sequential(backbone.Linear_fw(self.feat_dim, 128), backbone.BatchNorm1d_fw(128, track_running_stats=False))
    self.gnn = GNN_nl(128 + self.n_way, 96, self.n_way)
    self.method = 'fullmodel'

    # fix label for training the metric function   1*nw(1 + ns)*nw
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
    support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, n_way)], dim=1)
    self.support_label = support_label.view(1, -1, self.n_way)


  def cuda(self):
    self.feature.cuda()
    self.disentangle_model.cuda()
    self.domain_model.cuda()
    self.fc.cuda()
    self.gnn.cuda()
    self.support_label = self.support_label.cuda()
    return self


  def set_forward(self,x,is_feature=False):
    x = x.cuda()

    if is_feature:
      # reshape the feature tensor: n_way * n_s + 15 * f
      assert(x.size(1) == self.n_support + 15)
      z = self.fc(x.view(-1, *x.size()[2:]))
      z = z.view(self.n_way, -1, z.size(1))
    else:
      # get feature using encoder
      x = x.view(-1, *x.size()[2:])
      z = self.fc(self.feature(x))
      z = z.view(self.n_way, -1, z.size(1))
    # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores = self.forward_gnn(z_stack)
    return scores


  def forward_gnn(self, zs):
    # gnn inp: n_q * n_way(n_s + 1) * f
    nodes = torch.cat([torch.cat([z, self.support_label], dim=2) for z in zs], dim=0)
    scores = self.gnn(nodes)

    # n_q * n_way(n_s + 1) * n_way -> (n_way * n_q) * n_way
    scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
    return scores

  def set_forward_loss_init(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()
    scores = self.set_forward(x)
    loss = self.loss_fn(scores, y_query)
    return scores, loss

  def set_forward_loss_for_test(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()
    x_fea = self.set_forward_feature_extractor(x)
    a_code, b_code = self.disentangle_model(x_fea)
    z = self.fc(a_code)
    z = z.view(self.n_way, -1, z.size(1))
    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    scores = self.forward_gnn(z_stack)
    loss = self.loss_fn(scores, y_query)
    return scores, loss


  def set_forward_feature_extractor(self, x):
        x = x.cuda()
        x = x.view(-1, *x.size()[2:])
        fea = self.feature(x)
        return fea 

  def set_forward_disentangle_module(self, x_fea):
        a_fea, b_fea = self.disentangle_model(x_fea)
        return a_fea, b_fea 


  def set_forward_FSL_classifier(self, x):
      z = self.fc(x)
      z = z.view(self.n_way, -1, z.size(1))
      z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
      assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
      scores = self.forward_gnn(z_stack)
      return scores


  def set_forward_loss_for_train(self, x_1, x_2):
        # get support and query
        x_1_support = x_1[:,:self.n_support,:,:,:]
        x_1_query = x_1[:,self.n_support:,:,:,:]

        x_2_support = x_2[:,:self.n_support,:,:,:]
        x_2_query = x_2[:,self.n_support:,:,:,:]

        # mix the query
        mixed_query, lamda = mixup_data(x_1_query, x_2_query)

 
        # forward feature_extractor
        x_1_S_fea = self.set_forward_feature_extractor(x_1_support)
        x_2_S_fea = self.set_forward_feature_extractor(x_2_support)
        mix_Q_fea = self.set_forward_feature_extractor(mixed_query)


        # forward disentangle module
        input_fea_concat_1 = torch.cat((x_1_S_fea, x_2_S_fea), dim=0)
        input_fea_concat = torch.cat((input_fea_concat_1, mix_Q_fea), dim=0)
        a_code, b_code= self.set_forward_disentangle_module(input_fea_concat)

        x_1_S_len = x_1_S_fea.size()[0]
        x_2_S_len = x_2_S_fea.size()[0]
        mix_Q_len = mix_Q_fea.size()[0]
        x_1_S_a_code, x_1_S_b_code = a_code[0:x_1_S_len, :], b_code[0:x_1_S_len, :]
        x_2_S_a_code, x_2_S_b_code = a_code[x_1_S_len: x_1_S_len+x_2_S_len, :], b_code[x_1_S_len: x_1_S_len+x_2_S_len, :]
        mix_Q_a_code, mix_Q_b_code = a_code[x_1_S_len + x_2_S_len:, :], b_code[x_1_S_len + x_2_S_len:, :]


        # forward FSL classifier --> score,   (the domain-irrelevant a code is used)
        x_1_F, x_2_F, mix_F = x_1_S_a_code, x_2_S_a_code, mix_Q_a_code
        x_1_F = x_1_F.view(self.n_way, -1, x_1_F.size()[1])
        x_2_F = x_2_F.view(self.n_way, -1, x_2_F.size()[1])
        mix_F = mix_F.view(self.n_way, -1, mix_F.size()[1])
        mixup_x_1 = torch.cat((x_1_F, mix_F), 1)
        mixup_x_2 = torch.cat((x_2_F, mix_F), 1)
        mixup_x_1 = mixup_x_1.view(-1, mixup_x_1.size()[2])
        mixup_x_2 = mixup_x_2.view(-1, mixup_x_2.size()[2])
        scores_FSL_1 = self.set_forward_FSL_classifier(mixup_x_1)
        scores_FSL_2 = self.set_forward_FSL_classifier(mixup_x_2)

        # ground-truth for FSL classification
        y_query_1 = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
        y_query_2 = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
        y_query_1 = y_query_1.cuda()
        y_query_2 = y_query_2.cuda()

        # calculate the loss_FSL
        loss_FSL_1 = self.loss_fn(scores_FSL_1, y_query_1)
        loss_FSL_2 = self.loss_fn(scores_FSL_2, y_query_2)
        loss_FSL = lamda*loss_FSL_1 + (1-lamda)*loss_FSL_2

      
        # forward domain_classifier --> scores 
        x_1_S_a_domain_scores = self.domain_model(x_1_S_a_code)
        x_1_S_b_domain_scores = self.domain_model(x_1_S_b_code)
        x_2_S_a_domain_scores = self.domain_model(x_2_S_a_code)
        x_2_S_b_domain_scores = self.domain_model(x_2_S_b_code)
        mix_Q_a_domain_scores = self.domain_model(mix_Q_a_code)
        mix_Q_b_domain_scores = self.domain_model(mix_Q_b_code)


        # ground-truth for domain classification
        episode_batch = x_1_S_a_domain_scores.size()[0]
        y_1_S_a = Variable(torch.ones(episode_batch, 2)/2.0).cuda()   #[0,5, 0.5]
        y_1_S_b = Variable(torch.ones(episode_batch).long()).cuda()   #[1.0, 1.0]                         
        y_2_S_a = Variable(torch.ones(episode_batch, 2)/2.0).cuda()   #[0.5, 0.5]
        y_2_S_b = Variable(torch.zeros(episode_batch).long()).cuda()  #[0.0, 0.0]

        episode_batch_mix = mix_Q_a_domain_scores.size()[0]
        y_mix_a = Variable(torch.ones(episode_batch_mix, 2)/2.0).cuda()     #[0.5,0.5]
        y_mix_b_1 = Variable(torch.ones(episode_batch_mix).long()).cuda()   #[1.0,1.0] with a ratio of lamda
        y_mix_b_2 = Variable(torch.zeros(episode_batch_mix).long()).cuda()  #[0.0,0.0] with a ratio of (1-lamda)

        # calculate loss_domain_fusion (domain-irrelevant a code)
        loss_domain_fusion_1   = self.loss_KLD(F.log_softmax(x_1_S_a_domain_scores, dim=1), y_1_S_a)
        loss_domain_fusion_2   = self.loss_KLD(F.log_softmax(x_2_S_a_domain_scores, dim=1), y_2_S_a)
        loss_domain_fusion_mix = self.loss_KLD(F.log_softmax(mix_Q_a_domain_scores, dim=1), y_mix_a)
        loss_domain_fusion = (loss_domain_fusion_1 + loss_domain_fusion_2 + loss_domain_fusion_mix)/3.0
              
        # calculate loss domain_cls (domain-specific b code)
        loss_domain_CLS_1   = self.loss_fn(x_1_S_b_domain_scores, y_1_S_b) 
        loss_domain_CLS_2   = self.loss_fn(x_2_S_b_domain_scores, y_2_S_b)
        loss_domain_CLS_mix = lamda*self.loss_fn(mix_Q_b_domain_scores, y_mix_b_1) + (1-lamda)*self.loss_fn(mix_Q_b_domain_scores, y_mix_b_2)
        loss_domain_CLS = (loss_domain_CLS_1 + loss_domain_CLS_2 + loss_domain_CLS_mix)/3.0 
        return loss_FSL, loss_domain_fusion, loss_domain_CLS


