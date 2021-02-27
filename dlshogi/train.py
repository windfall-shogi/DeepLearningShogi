#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import Any, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
# noinspection PyProtectedMember
from torch.utils.data import DataLoader, Dataset
from torch.optim.swa_utils import AveragedModel, SWALR

import cshogi
import cppshogi
from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
from dlshogi.network.policy_value import PolicyValueNetwork
from dlshogi.pretrain import FeatureNetwork

__author__ = 'Yasuhiro'
__date__ = '2021/02/21'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--pattern', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=1000,
                        help='frequency (steps) to check validation')
    parser.add_argument('--block', type=int, default=20)
    parser.add_argument('--ch', type=int, default=256)
    parser.add_argument('--pre_act', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--swa_freq', type=int, default=250)
    parser.add_argument('--swa_n_avr', type=int, default=10)
    parser.add_argument('--swa_lr', type=float, default=1e-3)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--fast_dev_run', action='store_true')
    args = parser.parse_args()
    return args


def load_pretraining_model(model_path):
    model = FeatureNetwork.load_from_checkpoint(model_path)
    return model


def copy_pretrained_value(pretrained_model_path, model):
    """
    学習済みのモデルのパラメータをコピーする
    モデルの `base` の部分が学習済みモデルと共通でそれをコピー
    """
    pretrained = load_pretraining_model(pretrained_model_path)
    model.base.load_state_dict(pretrained.base.state_dict())


class Network(pl.LightningModule):
    # noinspection PyUnusedLocal
    def __init__(self, blocks, channels, features, pre_act=False,
                 activation=nn.SiLU, beta=0, val_lambda=0.333, lr=1e-2,
                 swa_freq=250):
        super(Network, self).__init__()
        self.save_hyperparameters()

        self.net = PolicyValueNetwork(blocks=blocks, channels=channels,
                                      features=features, pre_act=pre_act,
                                      activation=activation)
        self.swa_model = AveragedModel(self.net)

        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss()

    def load_pretrained_value(self, pretrained_model_path):
        copy_pretrained_value(pretrained_model_path=pretrained_model_path,
                              model=self.net.base)

    def forward(self, x):
        policy, value = self.net(x)
        return policy, value

    def training_step(self, batch, batch_idx):
        x1, x2, t1, t2, z, value = batch
        y1, y2 = self((x1, x2))

        t1 = t1.view(-1)

        loss1 = (self.ce(input=y1, target=t1) * z).mean()
        if self.hparams.beta > 0:
            entropy = F.softmax(y1, dim=1) * F.log_softmax(y1, dim=1)
            loss1 += self.hparams.beta * entropy.sum(dim=1).mean()
        loss2 = self.bce(input=y2, target=t2)
        loss3 = self.bce(input=y2, target=value)
        loss = (loss1 + (1 - self.hparams.val_lambda) * loss2 +
                self.hparams.val_lambda * loss3)

        self.log_dict({
            'loss': loss, 'loss/1': loss1, 'loss/2': loss2, 'loss/3': loss3,
            'accuracy/1':
                (torch.max(y1, dim=1)[1] == t1).type(torch.float32).mean(),
            'accuracy/2':
                ((y2 >= 0) == (t2 >= 0.5)).type(torch.float32).mean()
        })

        return loss

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int,
                           dataloader_idx: int) -> None:
        if (batch_idx + 1) % self.hparams.swa_freq == 0:
            self.swa_model.update_parameters(self.net)
            self.swa_scheduler.step()

    def validation_step(self, batch, batch_idx):
        x1, x2, t1, t2, z, value = batch
        y1, y2 = self((x1, x2))

        t1 = t1.view(-1)

        loss1 = (self.ce(input=y1, target=t1) * z).mean()
        loss2 = self.bce(input=y2, target=t2)
        loss3 = self.bce(input=y2, target=value)
        loss = (loss1 + (1 - self.hparams.val_lambda) * loss2 +
                self.hparams.val_lambda * loss3)

        entropy1 = (-F.softmax(y1, dim=1) *
                    F.log_softmax(y1, dim=1)).sum(dim=1)
        p2 = y2.sigmoid()
        log1p_ey2 = F.softplus(y2)
        entropy2 = -(p2 * (y2 - log1p_ey2) + (1 - p2) * -log1p_ey2)

        result = {
            'val_loss': loss, 'val_loss/1': loss1, 'val_loss/2': loss2,
            'val_loss/3': loss3,
            'val_accuracy/1':
                (torch.max(y1, dim=1)[1] == t1).type(torch.float32).mean(),
            'val_accuracy/2':
                ((y2 >= 0) == (t2 >= 0.5)).type(torch.float32).mean(),
            'val_entropy/1': entropy1.mean(), 'val_entropy/2': entropy2.mean()
        }
        self.log_dict(result)

        return result

    def test_step(self, batch, batch_idx):
        x1, x2, t1, t2, z, value = batch
        y1, y2 = self.swa_model((x1, x2))

        t1 = t1.view(-1)

        loss1 = (self.ce(input=y1, target=t1) * z).mean()
        loss2 = self.bce(input=y2, target=t2)
        loss3 = self.bce(input=y2, target=value)
        loss = (loss1 + (1 - self.hparams.val_lambda) * loss2 +
                self.hparams.val_lambda * loss3)

        entropy1 = (-F.softmax(y1, dim=1) *
                    F.log_softmax(y1, dim=1)).sum(dim=1)
        p2 = y2.sigmoid()
        log1p_ey2 = F.softplus(y2)
        entropy2 = -(p2 * (y2 - log1p_ey2) + (1 - p2) * -log1p_ey2)

        result = {
            'test_loss': loss, 'test_loss/1': loss1, 'test_loss/2': loss2,
            'test_loss/3': loss3,
            'test_accuracy/1':
                (torch.max(y1, dim=1)[1] == t1).type(torch.float32).mean(),
            'test_accuracy/2':
                ((y2 >= 0) == (t2 >= 0.5)).type(torch.float32).mean(),
            'test_entropy/1': entropy1.mean(),
            'test_entropy/2': entropy2.mean()
        }
        self.log_dict(result)

        return result

    # noinspection PyAttributeOutsideInit
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        self.swa_scheduler = SWALR(optimizer, swa_lr=self.hparams.lr,
                                   anneal_strategy='linear', anneal_epochs=10)
        return optimizer


def load_data(data_dir, pattern):
    """
    正規表現にマッチしたファイル名のものだけを読み込む
    """
    r = re.compile(pattern)
    data_dir = Path(data_dir)

    data_list = []
    for path in data_dir.glob('*'):
        name = path.name
        m = r.search(name)
        if m is None:
            continue

        print(path)
        tmp = np.fromfile(path, cshogi.HuffmanCodedPosAndEval)
        data_list.append(tmp)
    data_list = np.concatenate(data_list)
    return data_list


class HCPEDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature1 = np.empty((FEATURES1_NUM, 9, 9), dtype=np.float32)
        feature2 = np.empty((FEATURES2_NUM, 9, 9), dtype=np.float32)
        move = np.empty(1, dtype=np.int32)
        result = np.empty(1, dtype=np.float32)
        value = np.empty_like(result)

        # 要素を普通に取り出すとnp.void型になってしまう
        cppshogi.hcpe_decode_with_value(
            self.data[idx:idx + 1], feature1, feature2, move, result, value
        )

        z = result - value + 0.5

        return feature1, feature2, np.int64(move), result, z, value


def update_bn(loader, model, device=None):
    """
    torch.optim.swa_utils.update_bnではデータの形が合わなくて、
    GPUに転送できないので、書き直した
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for inputs in loader:
        x1, x2, t1, t2, z, value = inputs
        if device is not None:
            x1 = x1.to(device)
            x2 = x2.to(device)

        model((x1, x2))

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


def main():
    args = parse_args()

    train_data = load_data(data_dir=args.data_dir, pattern=args.pattern)
    test_data = np.fromfile(args.test_data,
                            dtype=cshogi.HuffmanCodedPosAndEval)
    train_dataset = HCPEDataset(data=train_data)
    val_dataset = HCPEDataset(data=test_data[:args.batch_size * 10])
    test_dataset = HCPEDataset(data=test_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False)

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if args.model_path is not None:
        model = Network.load_from_checkpoint(args.model_path)
    else:
        model = Network(blocks=args.block, channels=args.ch, features=256,
                        pre_act=args.pre_act, swa_freq=args.swa_freq)
        if args.pretrained_model_path is not None:
            copy_pretrained_value(
                pretrained_model_path=args.pretrained_model_path,
                model=model.net
            )

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=str(output_dir), monitor='val_loss',
        filename='pl-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3, mode='min'
    )
    trainer = pl.Trainer(
        callbacks=[checkpoint], max_epochs=args.epoch, gpus=[0],
        default_root_dir=str(output_dir),
        fast_dev_run=args.fast_dev_run,
        precision=16 if args.use_amp else 32,
        val_check_interval=args.eval_interval
    )
    trainer.fit(model, train_dataloader=train_loader,
                val_dataloaders=val_loader)

    train_dataset2 = HCPEDataset(data=train_data[:100000])
    train_loader2 = DataLoader(
        train_dataset2, batch_size=args.batch_size, shuffle=True
    )
    device = torch.device('cuda:0')
    swa_model = model.swa_model.to(device=device)
    update_bn(train_loader2, swa_model, device=device)
    metrics = trainer.test(model, test_dataloaders=test_loader)
    if isinstance(metrics, list):
        metrics = metrics[-1]
    with (output_dir / 'result.txt').open('w') as f:
        for key, value in metrics.items():
            f.write('{}: {}\n'.format(key, value))


if __name__ == '__main__':
    main()
