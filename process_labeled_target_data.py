
import os
import json
import torch
import random
import numpy as np
from options import parse_args
from data.datamgr import SimpleDataManager


def random_sample_labeled_base_from_testset_base(params, testset, NUM_UNLABEL):
    testset_base_file = os.path.join(params.data_dir, testset, 'base.json')
    content_json = json.load(open(testset_base_file,'r'))
    image_names = content_json['image_names']
    image_labels = content_json['image_labels']

    # make sure that the imgs wil not appear in the test set 
    testset_novel_file = os.path.join(params.data_dir, testset,'novel.json')
    novel_content_json = json.load(open(testset_novel_file,'r'))
    novel_image_names = novel_content_json['image_names']
    novel_image_labels = novel_content_json['image_labels']
  
    
    for img in novel_image_names:
        if (img in image_names):
            print('attention!')

    
    selected_idx = []
    selected_img_names = []
    selected_img_labels = []

    tmp_class_name = 'null'
    tmp_global_class_id = 0
    for idx in range(len(image_names)):
        img = image_names[idx]
        class_name = img.split('/')[-2]
        if(class_name!=tmp_class_name):
            if(tmp_class_name!='null'):
                tmp_selected_idx = random.sample(tmp_class_idx_list, NUM_UNLABEL)
                for i in tmp_selected_idx:
                    selected_img_names.append(image_names[i])
                    selected_img_labels.append(tmp_global_class_id)
                tmp_global_class_id += 1
            tmp_class_name = class_name
            tmp_class_idx_list = []  
            tmp_class_idx_list.append(idx)
        else:
            tmp_class_idx_list.append(idx)
    

    label_base = {}
    label_base['image_names'] = selected_img_names
    label_base['image_labels'] = selected_img_labels   
    
    # write into the json
    jsObj = json.dumps(label_base)
    file_name = 'output/labled_base_'+str(testset)+'_'+str(NUM_LABEL) +'.json'
    fileObject = open(file_name, 'w')
    fileObject.write(jsObj)
    fileObject.close()
    print('base dict saved in', file_name)
    return file_name


def combine_two_base_file(file_1, file_2):
   print('file1:', file_1)
   print('file2:', file_2)

   content_json_1 = json.load(open(file_1,'r'))
   content_json_2 = json.load(open(file_2,'r'))
   
   image_names_1 = content_json_1['image_names']
   image_labels_1 = content_json_1['image_labels']

   image_names_2 = content_json_2['image_names']
   image_labels_2 = content_json_2['image_labels']
   
   print(len(image_names_1), image_names_1[0:10], image_names_1[-10:])
   print(len(image_labels_1), image_labels_1[0:10], image_labels_1[-10:])
   print(len(image_names_2), image_names_2[0:10], image_names_2[-10:])
   print(len(image_labels_2), image_labels_2[0:10], image_labels_2[-10:])


   for i in range(len(image_labels_2)):
       image_labels_2[i] = image_labels_2[i] + 64
  
   image_names = image_names_1 + image_names_2
   image_labels = image_labels_1  + image_labels_2
   

   con = {}
   con['image_names'] = image_names
   con['image_labels'] = image_labels

   # write into the json
   jsObj = json.dumps(con)
   file_name = 'output/com_miniImagenet_and_'+file_2.split('/')[1]
   fileObject = open(file_name, 'w')
   fileObject.write(jsObj)
   fileObject.close()
   print('base dict saved in', file_name)
   return file_name
   

def read_all_images(img_json_file,image_size):
   aug_flag = False
   batch_size = 16
   datamgr =  SimpleDataManager(image_size, batch_size)
   dataloader = datamgr.get_data_loader(img_json_file, aug = aug_flag)
   imgs = []
   for i, (x,y) in enumerate(dataloader):
       print('dataloader: i:', i, 'x:', x.size(), 'y:', y.size())
       for j in range(x.size()[0]):
           imgs.append(x[j])
   imgs = torch.stack(imgs)
   return imgs

if __name__ == '__main__':
   # 1, random sample num_target examples/class from target base
   testset = 'cub'
   #testset='cars'
   #testset = 'places'
   #testset = 'plantae'
   params = parse_args('train')
   
   NUM_LABEL = 5
   label_base_file = random_sample_labeled_base_from_testset_base(params, testset, NUM_LABEL)
   print('main: label_base_file:', label_base_file)

   
   # 2. combine data from mini_imagenet and target e.g., cub
   file_1 = '/DATACENTER/4/lovelyqian/CROSS-DOMAIN-FSL-DATASETS/miniImagenet/base.json'
   file_2 = 'output/labled_base_cub_5.json' 
   combine_two_base_file(file_1, file_2)

