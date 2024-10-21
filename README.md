This is pytorch implementation of FusionCount paper(https://arxiv.org/abs/2202.13660) + onnx export/run
<br>
To see the process, visit https://colab.research.google.com/drive/1xoyjf7d-t0-Gl71B24D8TvP5oXrhGyIf
<br>
<br>
For more information please refer to https://github.com/Yiming-M/FusionCount/tree/main
<br>
<br>
## Intro
Based on DM-Count(https://github.com/cvlab-stonybrook/DM-Count)
- added preprocessing // preprocess.py
- added onnx export // torchToOnnx.py
- added onnx run // runOnnx.py
- changed model architecture // models.py

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



