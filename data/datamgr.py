# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import json
import torch
import random
from PIL import Image
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset, SetDataset, MultiSetDataset, EpisodicBatchSampler, MultiEpisodicBatchSampler, RandomLabeledTargetDataset
from abc import abstractmethod

class TransformLoader:
  def __init__(self, image_size,
      normalize_param = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
    self.image_size = image_size
    self.normalize_param = normalize_param
    self.jitter_param = jitter_param

  def parse_transform(self, transform_type):
    if transform_type=='ImageJitter':
      method = add_transforms.ImageJitter( self.jitter_param )
      return method
    method = getattr(transforms, transform_type)

    if transform_type=='RandomResizedCrop':
      return method(self.image_size)
    elif transform_type=='CenterCrop':
      return method(self.image_size)
    elif transform_type=='Resize':
      return method([int(self.image_size*1.15), int(self.image_size*1.15)])
    elif transform_type=='Normalize':
      return method(**self.normalize_param )
    else:
      return method()

  def get_composed_transform(self, aug = False):
    if aug:
      transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
    else:
      transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']

    transform_funcs = [ self.parse_transform(x) for x in transform_list]
    transform = transforms.Compose(transform_funcs)
    return transform




# added by fuyuqian in 2021 0107
class LabeledTargetDataset:
   def __init__(self, data_file,image_size, batch_size = 16, aug=True):
       with open(data_file, 'r') as f:
           self.meta = json.load(f)
       print('len of labeled target data:', len(self.meta['image_names']))
       # define transform
       self.batch_size = batch_size
       self.trans_loader = TransformLoader(image_size)
       self.transform = self.trans_loader.get_composed_transform(aug)

   def get_epoch(self):
       # return random
       idx_list = [i for i in range(len(self.meta['image_names']))]
       selected_idx_list = random.sample(idx_list, self.batch_size)
       
       img_list = []
       img_label = []
     
       for idx in selected_idx_list:
           image_path = self.meta['image_names'][idx]
           image_label = self.meta['image_labels'][idx]
           img = Image.open(image_path).convert('RGB')
           img = self.transform(img)
           img_list.append(img)
           img_label.append(image_label)
       #print(img_label)
       img_list = torch.stack(img_list)
       #img_label = torch.stack(img_label)
       img_label = torch.LongTensor(img_label)
       #print('img_list:', img_list.size())
       #print('img_label:', img_label.size())
       return img_list, img_label



class DataManager:
  @abstractmethod
  def get_data_loader(self, data_file, aug):
    pass

class SimpleDataManager(DataManager):
  def __init__(self, image_size, batch_size):
    super(SimpleDataManager, self).__init__()
    self.batch_size = batch_size
    self.trans_loader = TransformLoader(image_size)

  def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    dataset = SimpleDataset(data_file, transform)
    data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

    return data_loader


# added in 20210108
class RandomLabeledTargetDataManager(DataManager):
  def __init__(self, image_size, batch_size):
    super(RandomLabeledTargetDataManager, self).__init__()
    self.batch_size = batch_size
    self.trans_loader = TransformLoader(image_size)

  def get_data_loader(self, data_file, data_file_miniImagenet, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    dataset = RandomLabeledTargetDataset(data_file, data_file_miniImagenet, transform)
    data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

    return data_loader

class SetDataManager(DataManager):
  def __init__(self, image_size, n_way, n_support, n_query, n_eposide=100):
    super(SetDataManager, self).__init__()
    self.image_size = image_size
    self.n_way = n_way
    self.batch_size = n_support + n_query
    self.n_eposide = n_eposide

    self.trans_loader = TransformLoader(image_size)

  def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    if isinstance(data_file, list):
      dataset = MultiSetDataset( data_file , self.batch_size, transform )
      sampler = MultiEpisodicBatchSampler(dataset.lens(), self.n_way, self.n_eposide )
    else:
      dataset = SetDataset( data_file , self.batch_size, transform )
      sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )
    data_loader_params = dict(batch_sampler = sampler,  num_workers=4)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader

'''

# added in 20210109
class RandomLabeledTargetSetDataManager(DataManager):
  def __init__(self, image_size, n_way, n_support, n_query, n_eposide=100):
    super(RandomLabeledTargetSetDataManager, self).__init__()
    self.image_size = image_size
    self.n_way = n_way
    self.batch_size = n_support + n_query
    self.n_eposide = n_eposide

    self.trans_loader = TransformLoader(image_size)

  def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    if isinstance(data_file, list):
      dataset = MultiSetDataset( data_file , self.batch_size, transform )
      sampler = MultiEpisodicBatchSampler(dataset.lens(), self.n_way, self.n_eposide )
    else:
      dataset = SetDataset( data_file , self.batch_size, transform )
      sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )
    data_loader_params = dict(batch_sampler = sampler,  num_workers=4)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader
 '''
