# coding=utf-8
import gc
import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from ultralytics import YOLO, RTDETR

seed = 42

# CUDNN Accelerate
cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = True

# SET Seed
torch.cuda.set_device(0)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# Clear Memory
torch.cuda.empty_cache()
gc.collect()

# Train
model = YOLO(r'ultralytics/cfg/models/improved/Attention/yolov8 - CBAM.yaml')
results = model.train(data=r'ultralytics/cfg/datasets/coco.yaml')

# predict
# model = YOLO(r' ')
# source=r' '
# model.predict(source, save=True)
