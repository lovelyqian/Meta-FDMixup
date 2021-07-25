# 1 Meta-FDMIxup

Repository for the paper :  **Meta-FDMixup: Cross-Domain Few-Shot Learning Guided byLabeled Target Data**

[to add paper link]()



![image](https://user-images.githubusercontent.com/49612387/126885825-72bdcec9-61b9-484e-a11f-5af812d0f6ba.png)

If you have any questions, feel free to contact me.  My email is fuyq20@fudan.edu.cn.


# 2 setup and datasets

## 2.1 setup

A anaconda envs is recommended:

```
conda create --name py36 python=3.6
conda activate py36
conda install pytorch torchvision -c pytorch
pip3 install scipy>=1.3.2
pip3 install tensorboardX>=1.4
pip3 install h5py>=2.9.0
```

Then, git clone our repo:
```
git clone https://github.com/lovelyqian/Meta-FDMixup
cd Meta-FDMixup
```

## 2.2 datasets
Totally five datasets inculding miniImagenet, CUB, Cars, Places, and Plantae are used.

1. Following [FWT-repo](https://github.com/hytseng0509/CrossDomainFewShot) to download and setup all datasets. (It can be done quickly)

2. Remember to modify your own dataset dir in the 'options.py'

3. Under our new setting, we randomly select $num_{target}$ labeled images from the target base set to form the auxiliary set. The splits we used are provided in 'Sources/'.



# 3 usage
## 3.1 network pretraining
```
python3 network_train.py --stage pretrain  --name pretrain-model --train_aug 
```

## 3.2 pretrained model testing
```
# test source dataset (miniImagenet)
python network_test.py --ckp_path output/checkpoints/pretrain-model/399.tar --stage pretrain --dataset miniImagenet --save_epoch 399 --n_shot 5 

# test target dataset e.g. cub
python network_test.py --ckp_path output/checkpoints/pretrain-model/399.tar --stage pretrain --dataset cub --save_epoch 399 --n_shot 5
```

## 3.3 network meta-training
```
# traget set: CUB
python3 network_train.py --stage metatrain --name metatrain-model-5shot-cub --train_aug --warmup output/checkpoints/pretrain-model/399.tar --target_set cub --n_shot 5

# target set: Cars
python3 network_train.py --stage metatrain --name metatrain-model-5shot-cars --train_aug --warmup output/checkpoints/pretrain-model/399.tar --target_set cars --n_shot 5

# target set: Places
python3 network_train.py --stage metatrain --name metatrain-model-5shot-places --train_aug --warmup output/checkpoints/pretrain-model/399.tar --target_set places --n_shot 5

# target set: Plantae
python3 network_train.py --stage metatrain --name metatrain-model-5shot-plantae --train_aug --warmup output/checkpoints/pretrain-model/399.tar --target_set plantae --n_shot 5
```

# 4 pretrained ckps
We will release our pretrained models.


# 5 citing
If you find our paper or this code useful for your research, please cite us:
```
to update
```

# 6 Note
Notably, our code is built upon the implementation of [FWT-repo](https://github.com/hytseng0509/CrossDomainFewShot). 
