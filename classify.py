import torch
from torch import nn
import numpy as np
from torchvision import models
from torchmetrics import Accuracy
from model import ViolenceClassifier
from dataset import CustomDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import os

class ViolenceClass:
    def __init__(self, num_classes=2, learning_rate=1e-3, batchsize = 128, gpu_id = None):
        # 加载模型、设置参数等 modified
        ckpt_root = "./"
        ckpt_path = ckpt_root + "train_logs/resnet18_pretrain_test/version_0/checkpoints/resnet18_pretrain_test-epoch=31-val_loss=0.04.ckpt"
        self.model = ViolenceClassifier.load_from_checkpoint(ckpt_path)
        self.model.eval()
        self.batchsize = batchsize
        self.gpu_id = gpu_id
    
    def misc(self):
        # 其他处理函数
        pass
        
    def classify(self, imgs : torch.Tensor) -> list:
        # 图像分类
        result = self.model(imgs)
        return result.argmax(dim=1).tolist()
    

    # 测试模型的预测精确度
    def test_acc(self):
        # 加载测试数据集
        data_module = CustomDataModule(batch_size=self.batchsize)


        # 设置日志文件夹名
        log_name = "resnet18_pretrain"

        # 实例化日志记录器
        logger = TensorBoardLogger("test_logs", name=log_name)

        # 实例化训练器并配置为GPU上运行
        trainer = Trainer(accelerator='gpu', devices=self.gpu_id)

        # 开始测试
        trainer.test(self.model, data_module) 

if __name__ == '__main__':
    from torchvision import transforms
    from PIL import Image
    # 实例化已训练好的模型
    model1 = ViolenceClass()

    # 测试模型效果
    # model1.test_acc()

    # 用模型对图像进行分类，并输出分类结果
    results = []
    for img_name in os.listdir('./test_images'):
        img_dir = os.path.join('./test_images', img_name)
        img = Image.open(img_dir)
        img_transform = transforms.Compose([transforms.ToTensor()])
        img = img_transform(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            result = model1.classify(img)
        print(result)
        exit()
        results.append(result)
        # probabilities = torch.sigmoid(result).tolist()
        # comparison_result = 1 if probabilities[0][0] < probabilities[0][1] else 0
        # results.append(comparison_result)

    # 统计1的数量
    num_ones = sum(results)

    # 计算1的比例
    proportion_of_ones = num_ones / len(results)

    print(f"Proportion of 1's: {proportion_of_ones}")


        