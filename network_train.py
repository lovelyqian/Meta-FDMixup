import os
import random
import numpy as np
import torch
import torch.optim


from data.datamgr import SimpleDataManager, SetDataManager
from methods import backbone
from methods.backbone import model_dict
from methods.baselinetrain import BaselineTrain
from methods.meta_FDMixup_model import MetaFDMixup
from utils import load_state_to_the_backbone
from options import parse_args, get_resume_file, load_warmup_state




def train(base_loader, val_loader, model, start_epoch, stop_epoch, params, labeled_target_loader=None):
  # get optimizer and checkpoint path
  if(params.stage=='pretrain'):
      optimizer = torch.optim.Adam(model.parameters())

  elif(params.stage=='metatrain'):
      lr_new = 0.002
      optimizer = torch.optim.Adam([{'params': model.feature.parameters()},
                                {'params': model.disentangle_model.fc1.parameters()},
                                {'params': model.disentangle_model.bn1.parameters()},
                                {'params': model.disentangle_model.fc21a.parameters()},
                                {'params': model.disentangle_model.fc22a.parameters()},

                                {'params': model.disentangle_model.fc21b.parameters(), 'lr': lr_new},
                                {'params': model.disentangle_model.fc22b.parameters(), 'lr': lr_new},
                                {'params': model.fc.parameters(), 'lr': lr_new},
                                {'params': model.domain_model.parameters(), 'lr': lr_new},
                                {'params': model.gnn.parameters(), 'lr': lr_new}], 
                                lr = 1e-3)

  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # for validation
  max_acc = 0
  total_it = 0

  # start
  for epoch in range(start_epoch,stop_epoch):
    model.train()
    if(params.stage=='pretrain'):
        total_it = model.train_loop(epoch, base_loader,  optimizer, total_it) #model are called by reference, no need to return
    elif(params.stage=='metatrain'):
        total_it = model.train_loop(epoch, base_loader, labeled_target_loader, optimizer, total_it) #model are called by reference, no need to return
    model.eval()

    acc = model.test_loop( val_loader)
    if acc > max_acc :
      print("best model! save...")
      max_acc = acc
      outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
      torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
    else:
      print("GG! best accuracy {:f}".format(max_acc))

    if ((epoch + 1) % params.save_freq==0) or (epoch==stop_epoch-1):
      outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
      torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

  return model


# --- main function ---
if __name__=='__main__':
  '''
  # set random seed
  seed = 0
  print("set seed = %d" % seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  '''

  # parser argument
  params = parse_args('train')
  print('--- baseline training: {} ---\n'.format(params.name))
  print(params)


  # output and tensorboard dir
  params.tf_dir = '%s/log/%s'%(params.save_dir, params.name)
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)


  # dataloader
  print('\n--- prepare source dataloader ---')
  source_base_file  = os.path.join(params.data_dir, 'miniImagenet', 'base.json')
  source_val_file   = os.path.join(params.data_dir, 'miniImagenet', 'val.json')

  # model
  print('\n--- build model ---')
  image_size = 224

  if params.stage == 'pretrain':
    print('  pre-training the model using only the miniImagenet source data')
    base_datamgr    = SimpleDataManager(image_size, batch_size=16)
    base_loader     = base_datamgr.get_data_loader(source_base_file , aug=params.train_aug )
    val_datamgr     = SimpleDataManager(image_size, batch_size=64)
    val_loader      = val_datamgr.get_data_loader(source_val_file, aug=False)

    model           = BaselineTrain(model_dict[params.model], params.num_classes, tf_path=params.tf_dir)


  elif params.stage == 'metatrain':
    print('  meta training the model using the miniImagenet data and the {} auxiliary data'.format(params.target_set))

    #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    n_query = max(1, int(16* params.test_n_way/params.train_n_way))

    train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot)
    base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
    base_loader             = base_datamgr.get_data_loader( source_base_file , aug = params.train_aug )

    test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot)
    val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
    val_loader              = val_datamgr.get_data_loader( source_val_file, aug = False)

    # define labeled target dataloader
    labeled_base_file_dict = {}
    labeled_base_file_dict['cub'] = 'sources/labled_base_cub_' + str(params.target_num_label)+'.json'
    labeled_base_file_dict['cars'] = 'sources/labled_base_cars_' + str(params.target_num_label)+'.json'
    labeled_base_file_dict['places'] = 'sources/labled_base_places_' + str(params.target_num_label)+'.json'
    labeled_base_file_dict['plantae'] = 'sources/labled_base_plantae_' + str(params.target_num_label)+'.json'
    labeled_base_file = labeled_base_file_dict[params.target_set]
    labeled_target_datamgr = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
    labeled_target_loader = labeled_target_datamgr.get_data_loader(labeled_base_file, aug = params.train_aug)

    model           = MetaFDMixup(model_dict[params.model], tf_path=params.tf_dir, **train_few_shot_params)

  else:
    raise ValueError('Unknown method')

  model = model.cuda()


  # load model
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch
  if params.resume != '':
    resume_file = get_resume_file('%s/checkpoints/%s'%(params.save_dir, params.resume), params.resume_epoch)
    if resume_file is not None:
      tmp = torch.load(resume_file)
      start_epoch = tmp['epoch']+1
      model.load_state_dict(tmp['state'])
      print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
  elif 'pretrain' not in params.stage:
    if params.warmup == 'gg3b0':
      raise Exception('Must provide the pre-trained feature encoder file using --warmup option!')
    
    state = torch.load(params.warmup)['state']
    print(state.keys())
    print('here')
    print(model.state_dict().keys())
    model.load_state_dict(state, False)
    #model.feature.load_state_dict(state, strict=False)
    #model.disentangle_model.load_state_dict(state, strict=False)

  # training
  print('\n--- start the training ---')
  if(params.stage=='pretrain'):
    model = train(base_loader, val_loader,  model, start_epoch, stop_epoch, params)
  else:
    model = train(base_loader, val_loader, model, start_epoch, stop_epoch, params, labeled_target_loader)
