#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import os
from pathlib import Path
import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
import cshogi
from cshogi import CSA
from tqdm import tqdm

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
from dlshogi.network.resnet import NetworkBase

__author__ = 'Yasuhiro'
__date__ = '2021/02/14'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--csa_dir', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()
    return args


class FeatureNetwork(pl.LightningModule):
    def __init__(self, blocks, channels, pre_act=False, activation=nn.SiLU):
        super(FeatureNetwork, self).__init__()
        self.save_hyperparameters()

        self.base = NetworkBase(
            blocks=blocks, channels=channels, pre_act=pre_act,
            activation=activation
        )

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels * 2, out_channels=channels,
                      kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=channels),
            activation(),
            nn.Conv2d(in_channels=channels, out_channels=channels // 4,
                      kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=channels // 4),
            activation(),
            # global average pooling
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        # どっちが先か
        self.task1 = nn.Linear(in_features=channels // 4, out_features=1)
        # この後勝ったか
        self.task2 = nn.Linear(in_features=channels // 4, out_features=1)

        self.criterion1 = torch.nn.BCEWithLogitsLoss()
        self.criterion2 = torch.nn.MSELoss()

        self.train_accuracy1 = pl.metrics.Accuracy()
        self.train_accuracy2 = pl.metrics.ExplainedVariance()
        self.val_accuracy1 = pl.metrics.Accuracy()
        self.val_accuracy2 = pl.metrics.ExplainedVariance()

    def forward(self, xs):
        xa, xb = xs
        ha = self.base(xa)
        hb = self.base(xb)
        h = torch.cat((ha, hb), dim=1)

        h = self.net(h)
        h = h.view((-1, self.hparams.channels // 4))

        y1 = self.task1(h)
        y2 = self.task2(h)

        return y1, y2

    def _shared_step(self, batch, batch_idx, name):
        xs, (y1, y2) = batch

        p1, p2 = self(xs)

        y1 = y1.view(-1, 1)
        y2 = y2.view(-1, 1).tanh_()
        loss1 = self.criterion1(input=p1, target=y1)
        loss2 = self.criterion2(input=p2, target=y2)

        if 'val' in name:
            metrics1, metrics2 = self.val_accuracy1, self.val_accuracy2
            name2 = 'val_accuracy'
        else:
            metrics1, metrics2 = self.train_accuracy1, self.train_accuracy2
            name2 = 'accuracy'
        a1 = metrics1(p1, y1)
        a2 = metrics1(p2, y2)

        loss = loss1 + loss2
        self.log_dict({
            '{}/task1'.format(name): loss1, '{}/task2'.format(name): loss2,
            '{}'.format(name): loss,
            '{}/task1'.format(name2): a1, '{}/task2'.format(name2): a2
        })
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch=batch, batch_idx=batch_idx, name='loss')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch=batch, batch_idx=batch_idx,
                                 name='val_loss')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


class PositionPairDataset(Dataset):
    def __init__(self, record_list):
        super(PositionPairDataset, self).__init__()
        self.record_list = record_list
        self.board = cshogi.Board()

    def __len__(self):
        return len(self.record_list)

    def __getitem__(self, idx):
        kif = self.record_list[idx]
        self.board.reset()

        n = len(kif.moves)
        while n <= 20:
            # 時々、やたらと短い棋譜がある
            # その場合は次の棋譜を使う
            if idx == len(self.record_list) - 1:
                idx = 0
            else:
                idx += 1
            kif = self.record_list[idx]
            n = len(kif.moves)

        i = random.randint(0, n - 9)
        j = i + random.choice((2, 4, 6, 8))
        early = 0.0
        if random.random() < 0.5:
            i, j = j, i
            early = 1.0

        count = 0
        feature1a, feature2a, feature1b, feature2b = None, None, None, None
        result = None
        for k, move in enumerate(kif.moves):
            if i == k:
                result, feature1a, feature1b = self.make_input_feature(kif.win)
                count += 1
            elif j == k:
                _, feature2a, feature2b = self.make_input_feature(kif.win)
                count += 1

            if count == 2:
                break
            self.board.push(move)

        early = np.array(early, dtype=np.float32)
        # {0, 0.5, 1}を{-1, 0, 1}に変換
        result = np.array(result * 2 - 1, dtype=np.float32)

        x = (feature1a, feature1b), (feature2a, feature2b)
        y = early, result
        return x, y

    def make_input_feature(self, result):
        feature1 = np.empty((FEATURES1_NUM, 9, 9), dtype=np.float32)
        feature2 = np.empty((FEATURES2_NUM, 9, 9), dtype=np.float32)

        r = self.board.convert_feature_with_result(result, feature1, feature2)
        return r, feature1, feature2


def load_records(record_dir):
    record_dir = Path(record_dir)
    record_list = []
    for csa_path in tqdm(record_dir.glob('*.csa'), desc='loading csa'):
        record_list.extend(CSA.Parser.parse_file(str(csa_path)))
    return record_list


def main():
    args = parse_args()

    record_list = load_records(args.csa_dir)
    train_dataset = PositionPairDataset(record_list=record_list[:-5000])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True)
    val_dataset = PositionPairDataset(record_list=record_list[-5000:])
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False)

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if args.model_path is not None:
        model = FeatureNetwork.load_from_checkpoint(args.model_path)
    else:
        model = FeatureNetwork(blocks=20, channels=256, pre_act=False)
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=str(output_dir), monitor='val_loss',
        filename='pl-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3, mode='min'
    )
    trainer = pl.Trainer(callbacks=[checkpoint], max_epochs=2, gpus=[0],
                         default_root_dir=str(output_dir))
    trainer.fit(model, train_dataloader=train_loader,
                val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
