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
from models import MLP

# metrics
from metrics import psnr

# opt
from opt import get_opts

# optimizers


seed_everything(1234, workers=True)  # 随机数生成器,暂时没用


class CordMLP(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        if hparams.arch == 'identity':
            self.net = MLP(n_in=2)
        self.loss = nn.MSELoss()  # 使用均方差作为损失函数
        self.training_step_outputs = []

    def forward(self, x):
        return self.net(x)

    def setup(self, stage=None):
        self.train_dataset = ImageDataset(self.hparams.image_path,
                                          split='train')
        self.val_dataset = ImageDataset(self.hparams.image_path,
                                        split='val')

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def configure_optimizers(self):
        self.optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)

        return self.optimizer

    def training_step(self, batch, batch_idx):
        RGB_predict = self(batch['uv'])

        loss = self.loss(RGB_predict, batch['rgb'])
        psnr_ = psnr(RGB_predict, batch['rgb'])

        self.log('train/loss', loss, prog_bar=True)  # 损失率
        self.log('train/psnr', psnr_, prog_bar=True)  # psnr可以设置进度条
        # self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        rgb_predicted = self(batch['uv'])

        loss = self.loss(rgb_predicted, batch['rgb'])
        psnr_ = psnr(rgb_predicted, batch['rgb'])

        log = {
            'val_loss': loss,
            'val_psnr': psnr_,
            'rgb_pred': rgb_predicted,
        }
        self.training_step_outputs.append(log)
        return log

    def create_function(self):
        has_run = [False]  # 使用列表来创建可变的状态

        def my_function():
            if has_run[0]:
                return
            has_run[0] = True
            print(self.training_step_outputs)
        return my_function

    # def validation_epoch_end(self, outputs):
    def on_validation_epoch_end(self):
        my_function = self.create_function()
        my_function()
        mean_loss = torch.stack([x['val_loss']
                                for x in self.training_step_outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr']
                                for x in self.training_step_outputs]).mean()
        rgb_pred = torch.cat([x['rgb_pred']
                             # (512*512, 3)
                              for x in self.training_step_outputs])
        rgb_pred = rearrange(rgb_pred, '(h w) c -> c h w',
                             h=2*self.train_dataset.r,
                             w=2*self.train_dataset.r)

        self.logger.experiment.add_image('val/image_pred',
                                         rgb_pred,
                                         self.global_step)
        self.training_step_outputs.clear()
        self.log('val/loss', mean_loss, prog_bar=True)
        self.log('val/psnr', mean_psnr, prog_bar=True)
        """
        mean_loss = torch.stack([x['val_loss']
                                 for x in self.training_step_outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr']
                                 for x in self.training_step_outputs]).mean()
        rgb_predicted = torch.cat([x['rgb_predicted']
                                   for x in self.training_step_outputs
                                   ])
        rgb_predicted = rearrange(rgb_predicted, '(h w) c -> h w c',
                                  h=2*self.train_dataset.r,
                                  w=2*self.train_dataset.r)
        self.logger.experiment.add_image('val/rgb_predicted',
                                         rgb_predicted,
                                         self.current_epoch)
        self.training_step_outputs.clear()
        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr)
        """


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
