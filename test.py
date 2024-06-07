from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model import ViolenceClassifier
from dataset import CustomDataModule


if __name__ == '__main__':  
    gpu_id = 'auto'
    batch_size = 128
    log_name = "resnet18_pretrain"

    data_module = CustomDataModule(batch_size=batch_size)
    ckpt_root = "./"
    ckpt_path = ckpt_root + "train_logs/resnet18_pretrain_test/version_0/checkpoints/resnet18_pretrain_test-epoch=31-val_loss=0.04.ckpt"
    logger = TensorBoardLogger("test_logs", name=log_name)

    model = ViolenceClassifier.load_from_checkpoint(ckpt_path)
    trainer = Trainer(accelerator='cpu', devices=gpu_id)
    trainer.test(model, data_module) 