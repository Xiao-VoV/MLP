from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
import torch
from torch import nn
from einops import rearrange

# datasets
from torch.utils.data import DataLoader
from dataset import ImageDataset

# models
from model import MLP

# metrics
from metrics import psnr

# opt
from opt import get_opts

# optimizers


seed_everything(1234, workers=True)  # 随机数生成器


class CordMLP(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        if hparams.arch == 'identity':
            self.net = MLP()
        self.loss = nn.MSELoss()  # 使用均方差作为损失函数
        self.training_step_outputs = []

    # def prepare_data(self, x):
    #     return self.net(x)
    def forward(self, x):
        return self.net(x)

    def setup(self, stage=None):
        self.train_dataset = ImageDataset(
            self.hparams.image_path,
            self.hparams.img_wh,
            split='train')
        self.val_dateset = ImageDataset(self.hparams.image_path,
                                        self.hparams.img_wh,
                                        split='val')

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

    def training_step(self, batch, batch_idx):
        RGB_predict = self.net(batch['uv'])
        loss = self.loss(RGB_predict, batch['rgb'])
        self.log('train/loss', loss)  # 损失率
        psnr_ = psnr(RGB_predict, batch['rgb'])
        self.log('train/psnr', psnr_,)  # psnr可以设置进度条
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        rgb_predicted = self(batch['uv'])

        loss = self.loss(rgb_predicted, batch['rgb'])
        psnr_ = psnr(rgb_predicted, batch['rgb'])

        log = {
            'val_loss': loss,
            'val_psnr': psnr_,
            'rgb_predicted': rgb_predicted,
        }

        return log

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average)

        self.training_step_outputs.clear()  # free memory
        # mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # mean_psnr = torch.stack([x['tval_loss'] for x in outputs]).mean()

        # rgb_predicted = torch.cat([x['rgb_predicted']
        #                            for x in outputs])  # 512*512*3

        # rgb_predicted = rearrange(rgb_predicted, '(h w) c -> h w c', h=512)

        # self.logger.experiment.add_image(
        #     'val/iamge_predicted', rgb_predicted, self.global_step)

        # self.log('val/loss', mean_loss, prog_bar=True)
        # self.log('val/psnr', mean_psnr, prog_bar=True)


if __name__ == '__main__':
    hparams = get_opts()
    system = CordMLP(hparams)

    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [pbar]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=True,
                      accelerator='auto',
                      devices=1,
                      num_sanity_val_steps=0,
                      log_every_n_steps=1,
                      check_val_every_n_epoch=20,
                      benchmark=True)

    trainer.fit(system)
