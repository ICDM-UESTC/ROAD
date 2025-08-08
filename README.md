# ROAD: Robust and Adaptive Denoising representation learning Framework for Multi-Modal Knowledge Graph Completion  
This repo provides an official implementation of ROAD as described in the paper:Robust and Adaptive Denoising representation learning Framework for Multi-Modal Knowledge Graph Completion
## overview
![model](./Road.png)
## Code Structure  
├── checkpoint  
│   ├── DB15K  
│   ├── MKG-W  
│   └── MKG-Y  
├── datasets  
│   ├── DB15K  
│   ├── MKG-W  
│   └── MKG-Y  
├── layers  
│   ├── __init__.py  
│   ├── layer.py  
├── models  
│   ├── __init__.py  
│   ├── model.py  
│   ├── modules.py  
│   ├── MoE.py  
│   └── ROAD.py  
├── ROAD.yml  
├── run.sh  
├── train.py  
├── utils    
|   ├── data_loader.py  
|   ├── data_util.py  
|   └── __init__.py  
## Dependency
You can run this command in the terminal from the project directory to create the required Python environment for the model.
` conda env create -f ROAD.yml -n ROAD `
## Train 
Then the following commands can be used to train and test our Modal.
  `nohup bash run.sh > db15k.log 2>&1 &`  
  `nohup bash run.sh > mkgw.log 2>&1 &`  
  `nohup bash run.sh > mkgy.log 2>&1 &`  

