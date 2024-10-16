import torch
from torch.nn import functional as F
from torch import nn

# datasets
from torch.utils.data import DataLoader
from dataset import ImageDataset

# models
from model import MLP


# optimizers
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

seed_everything(1234, workers=True)  # 随机数生成器


class CordMLP(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        if hparams.arch == 'identity':
            self.net = MLP()
        self.loss = nn.MSELoss()  # 使用均方差作为损失函数

    def prepare_data(self, x):
        return self.net(x)

    def setup(self, stage=None):
        self.train_dataset = ImageDataset(
            self.hparams.image_apth, split='train')
        self.val_dateset = ImageDataset(self.hparams.image_apth, split='val')

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dateset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def configure_optimizers(self):
        self.opt = Adam(self.net.parameters(), lr=self.hparams.lr)
        return [self.opt]

    def train_setup(self, batch, batch_idx):
        RGB_predict = self.net(batch['uv'])
        loss = self.loss(RGB_predict, batch['rgb'])
        self.log('train_loss', loss)  # 损失率

        return loss
