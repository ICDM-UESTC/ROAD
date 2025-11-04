# (ROAD)Unveiling Cross-Modal Consistency: Taming Inter- and Intra-Modal Noise for Robust Multi-Modal Knowledge Graph Completion

This repo provides an official implementation of **ROAD** as described in the paper:Robust and Adaptive Denoising representation learning Framework for Multi-Modal Knowledge Graph Completion

# Code Structure

```sh
ROAD
├─ datasets
│  ├─ DB15K
│  ├─ MKG-W
│  └─ MKG-Y
├─ layers
│  ├─ init.py
│  └─ layer.py
├─ models
│  ├─ init.py
│  ├─ model.py
│  ├─ modules.py
│  ├─ MoE.py
│  └─ ROAD.py
├─ utils
│  ├─ data_loader.py
│  ├─ data_util.py
│  └─ init.py
├─ ROAD.yml
├─ README.md
├─ run.sh
└─ train.py
```

#  Data

The structural data of the knowledge graph has been stored in the datasets/DB15K, datasets/MKG-W, and datasets/MKG-Y directories. The textual and visual data for the three datasets can be obtained from following links.
## DB15K
[text_features.pth](https://drive.google.com/uc?export=download&id=1p8_5MMU9JSvwNoUkcE59nkMiysULWWWF)  
[img_features.pth](https://drive.google.com/uc?export=download&id=1GUQo3IPT2olRtbMaJSBtnT50fT01cogE)  
## MKG-W  
[text_features.pth](https://drive.google.com/uc?export=download&id=1itDPa0-IDMLmjT-o3VZ9uGwxGbvE9t7x)  
[img_features.pth](https://drive.google.com/uc?export=download&id=1pG59gJfabMfPZvehnfbCdJQR85WxByl3)  
## MKG-Y  
[text_features.pth](https://drive.google.com/uc?export=download&id=1cxX73cZhCz7FzgqMxJBh7uamibMSf6uh)  
[img_features.pth](https://drive.google.com/uc?export=download&id=15quf8XLVaJ0VbZkx8QGIFYkvmHWepo_T)  

## Dependency
You can run this command in the terminal from the project directory to create the required Python environment for the model.  

```sh
conda env create -f ROAD.yml -n ROAD
```

```sh
 conda activate ROAD
```



# Train 

Then the following commands can be used to train our modal. Each command is configured with the hyperparameters that achieved the best performance reported in the paper.     


**DB15K**

```sh
nohup python -u train.py --cuda 0 --lr 0.001 --eval_freq 100 --dim 200 --dataset DB15K --epochs 2000 --beta_s 1e-5 --beta_t 1e-5 --beta_i 1e-5 --lamda_conf 1e-3 --lamda_cl 5e-5  > db15k.txt
```

**MKG-W**

```sh
nohup python -u train.py --cuda 0 --lr 0.001 --eval_freq 100 --dim 200 --dataset MKG-W --epochs 2000 --beta_s 1e-4 --beta_t 1e-4 --beta_i 1e-4 --lamda_conf 1e-4 --lamda_cl 1e-4  > mkgw.txt 
```

**MKG-Y**

```sh
nohup python -u train.py --cuda 0 --lr 0.001 --eval_freq 100 --dim 200 --dataset MKG-Y --epochs 2000 --beta_s 1e-3 --beta_t 1e-3 --beta_i 1e-3 --lamda_conf 1e-3 --lamda_cl 1e-3  > mkgy.txt
```


 






