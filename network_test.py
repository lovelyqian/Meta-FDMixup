import os
import random
import h5py
import torch
import numpy as np

from methods import backbone
from methods.backbone import model_dict
from methods.protonet import ProtoNet
from methods.disentangle_module import Disentangle
from methods.meta_FDMixup_model import MetaFDMixup

from data.datamgr import SimpleDataManager
import data.feature_loader as feat_loader
from options import parse_args, get_best_file, get_assigned_file


# extract and save image features
def save_features(model, model_disentangle, data_loader, featurefile):
  f = h5py.File(featurefile, 'w')
  max_count = len(data_loader)*data_loader.batch_size
  all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
  all_feats=None
  count=0
  for i, (x,y) in enumerate(data_loader):
    if (i % 10) == 0:
      print('    {:d}/{:d}'.format(i, len(data_loader)))
    x = x.cuda()
    feats = model(x)


    a_code, b_code  = model_disentangle(feats)
    feats = a_code
 

    if all_feats is None:
      all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
    all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
    all_labels[count:count+feats.size(0)] = y.cpu().numpy()
    count = count + feats.size(0)

  count_var = f.create_dataset('count', (1,), dtype='i')
  count_var[0] = count
  f.close()

# evaluate using features
def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15):
  class_list = cl_data_file.keys()
  select_class = random.sample(class_list,n_way)
  z_all  = []
  for cl in select_class:
    img_feat = cl_data_file[cl]
    perm_ids = np.random.permutation(len(img_feat)).tolist()
    z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )
  z_all = torch.from_numpy(np.array(z_all) )

  model.n_query = n_query
  scores  = model.set_forward(z_all, is_feature = True)
  pred = scores.data.cpu().numpy().argmax(axis = 1)
  y = np.repeat(range( n_way ), n_query )
  acc = np.mean(pred == y)*100
  return acc



def test_single_ckp(params):
  print('\nStage 1: saving features')
  # dataset
  print('  build dataset')
  image_size = 224
  split = params.split
  loadfile = os.path.join(params.data_dir, params.dataset, split + '.json')
  print('load file:', loadfile)
  datamgr         = SimpleDataManager(image_size, batch_size = 64)
  data_loader      = datamgr.get_data_loader(loadfile, aug = False)

  print('  build feature encoder')
  model = model_dict[params.model]()
  print('model:', model.state_dict().keys())
  model_disentangle = Disentangle()
  print('model disentangle:', model_disentangle.state_dict().keys())
  model = model.cuda()
  model_disentangle = model_disentangle.cuda()  
  tmp = torch.load(params.ckp_path)
  try:
    state = tmp['state']
  except KeyError:
    state = tmp['model_state']
  except:
    raise
  state_keys = list(state.keys())
  print('state_keys:', state_keys, len(state_keys))


  # load params to the backbone model
  state_backbone = state
  for i, key in enumerate(state_keys):
    if "feature." in key and not 'gamma' in key and not 'beta' in key:
      newkey = key.replace("feature.","")
      state_backbone[newkey] = state_backbone.pop(key)
    else:
      state_backbone.pop(key)
  print('backbone state keys:', list(state_backbone.keys()), len(list(state_backbone.keys())))
  model.load_state_dict(state_backbone)
  model.eval()
  
  # load prams to the disentangle model
  tmp = torch.load(params.ckp_path)
  try:  
    state = tmp['state'] 
  except KeyError:
    state = tmp['model_state']
  state_disentangle = state
  print('\n')
  print('state_disentangle:', state_disentangle.keys())
  for i,key in enumerate(state_keys):
      if ("disentangle_model." in key):
          newkey = key.replace("disentangle_model.", "")
          state_disentangle[newkey] = state_disentangle.pop(key)
      else:
          state_disentangle.pop(key)
  print('disentangle state keys:', list(state_disentangle.keys()), len(list(state_disentangle.keys())))
  model_disentangle.load_state_dict(state_disentangle)
  model_disentangle.eval()
  print('len check:', len(list(state_backbone.keys())) + len(list(state_disentangle.keys())), len(state_keys))

  # save feature file
  print('  extract and save features...')
  featurefile = params.ckp_path.replace('checkpoints','features').replace('tar', 'hdf5')
  dirname = os.path.dirname(featurefile)
  if not os.path.isdir(dirname):
    os.makedirs(dirname)
  save_features(model, model_disentangle, data_loader, featurefile)



  print('\nStage 2: evaluate')
  acc_all = []
  iter_num = 1000
  few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)
  # model
  print('  build metric-based model')
  if params.stage == 'pretrain':
    model = ProtoNet( model_dict[params.model], **few_shot_params)
  elif params.stage == 'metatrain':
    model = MetaFDMixup( model_dict[params.model], **few_shot_params)
    # load model
    tmp = torch.load(params.ckp_path)
    model.load_state_dict(tmp['state'])

  model = model.cuda()
  model.eval()


  # load feature file
  print('  load saved feature file')
  cl_data_file = feat_loader.init_loader(featurefile)

  # start evaluate
  print('  evaluate')
  for i in range(iter_num):
    acc = feature_evaluation(cl_data_file, model, n_query=15, **few_shot_params)
    acc_all.append(acc)

  # statics
  print('  get statics')
  acc_all = np.asarray(acc_all)
  acc_mean = np.mean(acc_all)
  acc_std = np.std(acc_all)
  print('  %d test iterations: Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

  # remove feature files [optional]
  if remove_featurefile:
    os.remove(featurefile)





# --- main ---
if __name__ == '__main__':
  # random seed
  '''
  seed = 0
  print("set seed = %d" % seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  '''

  # parse argument
  params = parse_args('test')
  print('Testing! {} shots on {} dataset with ckp:{})'.format(params.n_shot, params.dataset, params.ckp_path))
  remove_featurefile = True

  acc = test_single_ckp(params)


