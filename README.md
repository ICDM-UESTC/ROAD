# ROAD: Robust and Adaptive Denoising representation learning Framework for Multi-Modal Knowledge Graph Completion  
This repo provides an official implementation of ROAD as described in the paper:Robust and Adaptive Denoising representation learning Framework for Multi-Modal Knowledge Graph Completion
## overview
![model](./Road.png)
## Code Structure

```
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
├── utils    
│   ├── data_loader.py  
│   ├── data_util.py  
│   └── __init__.py  
├── ROAD.yml  
├── run.sh  
├── train.py  
```
## Data
The structural data of the knowledge graph has been stored in the datasets/DB15K, datasets/MKG-W, and datasets/MKG-Y directories. The textual and image data for the three datasets can be obtained from [Google Drive](https://drive.google.com/drive/folders/1C1E0lwRdgMlyDevEVB4Ri3-rknBhWnym?usp=drive_link).
## Dependency
You can run this command in the terminal from the project directory to create the required Python environment for the model.  
` conda env create -f ROAD.yml -n ROAD `
## Train 
Then the following commands can be used to train our Modal.    
DB15K  
  `nohup bash run.sh > db15k.log 2>&1 &`  
MKG-W  
  `nohup bash run.sh > mkgw.log 2>&1 &`    
MKG-Y  
  `nohup bash run.sh > mkgy.log 2>&1 &`  

