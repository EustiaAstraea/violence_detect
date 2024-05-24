import torch
from torch import nn
import numpy as np
from torchvision import models
from torchmetrics import Accuracy
from model import ViolenceClassifier

class ViolenceClass:
    def __init__(self, num_classes=2, learning_rate=1e-3):
        # 加载模型、设置参数等
        ckpt_root = "C:/Users/28214/Desktop/violence_detect/"
        ckpt_path = ckpt_root + "resnet18_pretrain_test/version_0/checkpoints/resnet18_pretrain_test-epoch=xx-val_loss=xx.ckpt"
        self.model = ViolenceClassifier.load_from_checkpoint(ckpt_path)

    
    def misc(self):
        # 其他处理函数
        pass
        
    def classify(self, imgs : torch.Tensor) -> list:
        # 图像分类
        return self.model.forward(imgs)
