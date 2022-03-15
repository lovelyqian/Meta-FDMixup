# 1 Meta-FDMIxup

Repository for the paper :  

**Meta-FDMixup: Cross-Domain Few-Shot Learning Guided byLabeled Target Data.** (ACM MM 2021)

[paper](https://arxiv.org/abs/2107.11978)

[News! the representation video loaded in 2021/10/06 in Bilibili](https://www.bilibili.com/video/BV1xT4y1f7B6?spm_id_from=333.999.0.0)

[News! the representation video loaded in 2021/10/10 in Youtube](https://www.youtube.com/watch?v=G8Mlde4FpsU)


![image](https://user-images.githubusercontent.com/49612387/126885825-72bdcec9-61b9-484e-a11f-5af812d0f6ba.png)

**[News! The results of meta-FDMixup under only single source domain are provided in 2022/03/15!]**

By sampling both two episodes from source dataset, we can easily adapt the meta-FDMixup to the single source domain setting. Here, we provide the 5-way-K-shot results on the FWT's and BS-CDFSL's benchmarks with no target data. 



| **Method** 	| **mini-Imagenet** 	| **CUB** 	| **Cars** 	| **Places** 	| **Plantae** 	| **ChestX** 	| **ISIC** 	| **EuroSAT** 	| **CropDisease** 	|
|---	|---	|---	|---	|---	|---	|---	|---	|---	|---	|
| **GNN (1 shot)** 	| 60.77 +- 0.75 	| 45.69 +- 0.68 	| 31.79 +- 0.51 	| 53.10 +- 0.80 	| 35.60 +- 0.56 	| 22.00 +- 0.46 	| 32.02 +- 0.66 	| 63.69 +- 1.03 	| 64.48 +- 1.08 	|
| **meta-FDMixup (1 shot)** 	| 62.12 +- 0.76 	| 46.38 +- 0.68 	| 31.14 +- 0.51 	| 53.57 +- 0.75  	| 37.89 +- 0.58 	| 22.26 +- 0.45 	| 32.48 +- 0.64 	| 62.97 +- 1.01 	| 66.23 +- 1.03 	|
| **GNN (5 shot)** 	| 80.87 +- 0.56 	| 62.25 +- 0.65 	| 44.28 +- 0.63 	| 70.84 +- 0.65 	| 52.53 +- 0.59 	| 25.27 +- 0.46 	| 43.94 +- 0.67 	| 83.64 +- 0.77 	| 87.96 +- 0.67 	|
| **meta-FDMixup (5 shot)** 	| 81.07 +- 0.55  	| 64.71 +- 0.68 	| 41.30 +- 0.58 	| 73.42 +- 0.65 	| 54.62 +- 0.66 	| 24.52 +- 0.44 	| 44.28 +- 0.66 	| 80.48 +- 0.79 	| 87.27 +- 0.69 	|

At most cases, meta-FDMixup still outperforms the GNN base.



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


# 3 pretrained ckps
We provide several pretrained ckps.

You can download and put them in the 'output/pretrained_ckps/'

## 3.1 **pretrained model trained on the miniImagenet**
- [pretrained_model_399.tar](https://drive.google.com/file/d/1iYu3lvYDixVNPYjmyi0MON8-X3aRN4n2/view?usp=sharing)


## 3.2 **full model meta-trained on the target datasets**

Since our method is target-set specific, we have to train a model for each target dataset.

- [full_model_5shot_target_cub_399.tar](https://drive.google.com/file/d/1UpRWkvUZ4FqJx542SdhL2AY-s8LhYi9y/view?usp=sharing)

- [full_model_5shot_target_cars_399.tar](https://drive.google.com/file/d/1b_XUQBGuG2FYhKq_R9smxiBOLWvpFHhn/view?usp=sharing)

- [full_model_5shot_target_places_399.tar](https://drive.google.com/file/d/1RLN3PWbC9FjCGsL4cn_iYqMv-ER5_bo9/view?usp=sharing)

- [full_model_5shot_target_plantae_399.tar](https://drive.google.com/file/d/11S_NyQkY4VV9T7Fb46tAxYO7ZO5YoYYj/view?usp=sharing)

Notably, as we stated in the paper, we use the last checkpoint for target dataset, while the best model on the validation set of miniImagenet is used for miniImagenet. Here, we provide the model of 'miniImagenet|CUB' as an example.

- [full_model_5shot_target_cub_best_eval.tar](https://drive.google.com/file/d/1eUlyHA3dov37YOh6phE25RUmaU_vYMiX/view?usp=sharing)



# 4 usage
## 4.1 network pretraining
```
python3 network_train.py --stage pretrain  --name pretrain-model --train_aug 
```

If you have downloaded our [pretrained_model_399.tar](https://drive.google.com/file/d/1iYu3lvYDixVNPYjmyi0MON8-X3aRN4n2/view?usp=sharing), you can just skip this step.


## 4.2 pretrained model testing
```
# test source dataset (miniImagenet)
python network_test.py --ckp_path output/checkpoints/pretrain-model/399.tar --stage pretrain --dataset miniImagenet --n_shot 5 

# test target dataset e.g. cub
python network_test.py --ckp_path output/checkpoints/pretrain-model/399.tar --stage pretrain --dataset cub --n_shot 5
```

you can test our [pretrained_model_399.tar](https://drive.google.com/file/d/1iYu3lvYDixVNPYjmyi0MON8-X3aRN4n2/view?usp=sharing) in the same way:


```
# test source dataset (miniImagenet)
python network_test.py --ckp_path output/pretrained_ckps/pretrained_model_399.tar --stage pretrain --dataset miniImagenet --n_shot 5 


# test target dataset e.g. cub
python network_test.py --ckp_path output/pretrained_ckps/pretrained_model_399.tar --stage pretrain --dataset cub --n_shot 5
```




## 4.3 network meta-training

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

Also, you can use our [pretrained_model_399.tar](https://drive.google.com/file/d/1iYu3lvYDixVNPYjmyi0MON8-X3aRN4n2/view?usp=sharing) for warmup:

```
# traget set: CUB
python3 network_train.py --stage metatrain --name metatrain-model-5shot-cub --train_aug --warmup output/pretrained_ckps/pretrained_model_399.tar --target_set cub --n_shot 5
```


## 4.4 network testing

To test our provided full models:
```
# test target dataset (CUB)
python network_test.py --ckp_path output/pretrained_ckps/full_model_5shot_target_cub_399.tar --stage metatrain --dataset cub --n_shot 5 

# test target dataset (Cars)
python network_test.py --ckp_path output/pretrained_ckps/full_model_5shot_target_cars_399.tar --stage metatrain --dataset cars --n_shot 5 

# test target dataset (Places)
python network_test.py --ckp_path output/pretrained_ckps/full_model_5shot_target_places_399.tar --stage metatrain --dataset places --n_shot 5 

# test target dataset (Plantae)
python network_test.py --ckp_path output/pretrained_ckps/full_model_5shot_target_places_399.tar --stage metatrain --dataset plantae --n_shot 5 


# test source dataset (miniImagenet|CUB)
python network_test.py --ckp_path output/pretrained_ckps/full_model_5shot_target_cub_best_eval.tar --stage metatrain --dataset miniImagenet --n_shot 5 
```

To test your models, just modify the 'ckp-path'.


# 5 citing
If you find our paper or this code useful for your research, please cite us:
```
@article{fu2021meta,
  title={Meta-FDMixup: Cross-Domain Few-Shot Learning Guided by Labeled Target Data},
  author={Fu, Yuqian and Fu, Yanwei and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2107.11978},
  year={2021}
}
```

# 6 Note
Notably, our code is built upon the implementation of [FWT-repo](https://github.com/hytseng0509/CrossDomainFewShot). 
