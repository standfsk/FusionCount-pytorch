This is unofficial implementation of FusionCount
- Paper: https://arxiv.org/abs/2202.13660
- Colab: https://colab.research.google.com/drive/1xoyjf7d-t0-Gl71B24D8TvP5oXrhGyIf
- Official implementation: https://github.com/Yiming-M/FusionCount/tree/main

## Intro
Based on DM-Count(https://github.com/cvlab-stonybrook/DM-Count)
- added preprocessing // preprocess.py
- added onnx export // torchToOnnx.py
- added onnx run // runOnnx.py
- changed model architecture // models.py

## Model
![161753152-1019e96e-18da-43de-9af0-46c6ed55bd12](https://github.com/user-attachments/assets/796d824f-130e-47cc-9136-d144c9591fee)
- Multi-scale fusion -> 4 fused features
- Channel reduction + upsample + fusing applied to 4 fused features respectively
- Loss function consists of three components
  - Counting Loss
  - Optimal Transport Loss
  - Total Variance Loss

## Installation
```
$ git clone https://github.com/standfsk/FusionCount-pytorch.git
$ cd FusionCount-pytorch
$ pip install -r requirements.txt
```

## Dataset
- NWPU(https://gjy3035.github.io/NWPU-Crowd-Sample-Code/)
- QNRF(https://www.crcv.ucf.edu/data/ucf-qnrf/)
- ShanghaiTech Part A/B(https://github.com/desenzhou/ShanghaiTechDataset)
- UCF-CC(https://www.crcv.ucf.edu/data/ucf-cc-50/UCFCrowdCountingDataset_CVPR13.rar)

## Preprocess
```
$ mkdir data
$ mkdir ckpt
$ python preprocess.py --dataset UCF-CC_50
```

## Train
```
$ python train.py --dataset UCF_CC_50 --save_path ckpt/ucfcc --batch-size 6 --max-epoch 100 --gpu_id 0
```

## Test
```
$ python test.py --weight_path ckpt/ucfcc/best_model.pth --dataset ucfcc --gpu_id 0
```

## Demo
```
$ python video_demo.py --weight_path ckpt/ucfcc/best_model.pth --video_name sample.mp4 --save --gpu_id 0 
```

## Result
![0009](https://github.com/user-attachments/assets/905dd24f-8ffb-4cd2-a0fc-5255124e732d)



