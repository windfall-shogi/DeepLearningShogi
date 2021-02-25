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
                 activation=nn.SiLU, beta=0, val_lambda=0.333, lr=1e-2):
        super(Network, self).__init__()
        self.save_hyperparameters()

        self.net = PolicyValueNetwork(blocks=blocks, channels=channels,
                                      features=features, pre_act=pre_act,
                                      activation=activation)

        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss()

        self.accuracy_list = nn.ModuleList((
            pl.metrics.Accuracy(), pl.metrics.Accuracy(),
            pl.metrics.Accuracy(), pl.metrics.Accuracy(),
            pl.metrics.Accuracy(), pl.metrics.Accuracy()
        ))
        self.train_metrics = {
            'loss1': 0, 'loss2': 0, 'loss3': 0, 'count': 0,
            'accuracy1': self.accuracy_list[0],
            'accuracy2': self.accuracy_list[1]
        }
        self.val_metrics = {
            'loss1': 0, 'loss2': 0, 'loss3': 0, 'count': 0,
            'accuracy1': self.accuracy_list[2],
            'accuracy2': self.accuracy_list[3],
            'entropy1': 0, 'entropy2': 0
        }
        self.test_metrics = {
            'loss1': 0, 'loss2': 0, 'loss3': 0, 'count': 0,
            'accuracy1': self.accuracy_list[4],
            'accuracy2': self.accuracy_list[5],
            'entropy1': 0, 'entropy2': 0
        }

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

        p1 = y1.softmax(dim=-1)
        p2 = y2.sigmoid()
        t2 = t2.type(torch.int64)

        self.train_metrics['loss1'] += loss1.item()
        self.train_metrics['loss2'] += loss2.item()
        self.train_metrics['loss3'] += loss3.item()
        self.train_metrics['accuracy1'].update(preds=p1, target=t1)
        self.train_metrics['accuracy2'].update(preds=p2, target=t2)
        self.train_metrics['count'] += 1

        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        loss1 = self.train_metrics['loss1'] / self.train_metrics['count']
        loss2 = self.train_metrics['loss2'] / self.train_metrics['count']
        loss3 = self.train_metrics['loss3'] / self.train_metrics['count']
        loss = (loss1 + (1 - self.hparams.val_lambda) * loss2 +
                self.hparams.val_lambda * loss3)
        self.log_dict({
            'loss': loss, 'loss/1': loss1, 'loss/2': loss2, 'loss/3': loss3,
            'accuracy/1': self.train_metrics['accuracy1'].compute(),
            'accuracy/2': self.train_metrics['accuracy2'].compute()
        })

        for key in self.train_metrics:
            if 'accuracy' in key:
                self.train_metrics[key].reset()
            else:
                self.train_metrics[key] = 0

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

        p1 = y1.softmax(dim=-1)
        t2 = t2.type(torch.int64)

        self.val_metrics['loss1'] += loss1.item()
        self.val_metrics['loss2'] += loss2.item()
        self.val_metrics['loss3'] += loss3.item()
        self.val_metrics['entropy1'] += entropy1.mean().item()
        self.val_metrics['entropy2'] += entropy2.mean().item()
        self.val_metrics['accuracy1'].update(preds=p1, target=t1)
        self.val_metrics['accuracy2'].update(preds=p2, target=t2)
        self.val_metrics['count'] += 1

        return loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        loss1 = self.val_metrics['loss1'] / self.val_metrics['count']
        loss2 = self.val_metrics['loss2'] / self.val_metrics['count']
        loss3 = self.val_metrics['loss3'] / self.val_metrics['count']
        loss = (loss1 + (1 - self.hparams.val_lambda) * loss2 +
                self.hparams.val_lambda * loss3)
        entropy1 = self.val_metrics['entropy1'] / self.val_metrics['count']
        entropy2 = self.val_metrics['entropy2'] / self.val_metrics['count']
        self.log_dict({
            'val_loss': loss, 'val_loss/1': loss1, 'val_loss/2': loss2,
            'val_loss/3': loss3,
            'val_accuracy/1': self.val_metrics['accuracy1'].compute(),
            'val_accuracy/2': self.val_metrics['accuracy2'].compute(),
            'val_entropy/1': entropy1, 'val_entropy/2': entropy2
        })

        for key in self.val_metrics:
            if 'accuracy' in key:
                self.val_metrics[key].reset()
            else:
                self.val_metrics[key] = 0

    def test_step(self, batch, batch_idx):
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

        p1 = y1.softmax(dim=-1)
        t2 = t2.type(torch.int64)

        self.test_metrics['loss1'] += loss1.item()
        self.test_metrics['loss2'] += loss2.item()
        self.test_metrics['loss3'] += loss3.item()
        self.test_metrics['entropy1'] += entropy1.mean().item()
        self.test_metrics['entropy2'] += entropy2.mean().item()
        self.test_metrics['accuracy1'].update(preds=p1, target=t1)
        self.test_metrics['accuracy2'].update(preds=p2, target=t2)
        self.test_metrics['count'] += 1

        result = {'loss': loss.item()}
        for key, value in self.test_metrics.items():
            if 'accuracy' in key:
                result[key] = value.compute()
            else:
                result[key] = value
        return result

    def test_epoch_end(self, outputs: List[Any]) -> None:
        loss1 = self.test_metrics['loss1'] / self.test_metrics['count']
        loss2 = self.test_metrics['loss2'] / self.test_metrics['count']
        loss3 = self.test_metrics['loss3'] / self.test_metrics['count']
        loss = (loss1 + (1 - self.hparams.val_lambda) * loss2 +
                self.hparams.val_lambda * loss3)
        entropy1 = self.test_metrics['entropy1'] / self.test_metrics['count']
        entropy2 = self.test_metrics['entropy2'] / self.test_metrics['count']
        self.log_dict({
            'test_loss': loss, 'test_loss/1': loss1, 'test_loss/2': loss2,
            'test_loss/3': loss3,
            'test_accuracy/1': self.test_metrics['accuracy1'].compute(),
            'test_accuracy/2': self.test_metrics['accuracy2'].compute(),
            'test_entropy/1': entropy1, 'test_entropy/2': entropy2
        })

    # noinspection PyAttributeOutsideInit
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
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


def main():
    args = parse_args()

    train_data = load_data(data_dir=args.data_dir, pattern=args.pattern)
    test_data = np.fromfile(args.test_data,
                            dtype=cshogi.HuffmanCodedPosAndEval)
    train_dataset = HCPEDataset(data=train_data)
    val_dataset = HCPEDataset(data=test_data[:1000])
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
        model = Network(blocks=20, channels=256, features=256,
                        pre_act=args.pre_act)
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
    swa = pl.callbacks.swa.StochasticWeightAveraging(
        swa_epoch_start=args.swa_freq, swa_lrs=args.lr
    )
    # 1 epochあたりのバッチ数を制限するので、全体のエポック数を調整
    n = args.swa_freq * args.batch_size
    epochs = (len(train_dataset) + n - 1) // n
    max_epochs = args.epoch * epochs
    print('max epochs:', max_epochs)
    # validation dataを評価する頻度も調整
    interval = (args.eval_interval + args.swa_freq - 1) // args.swa_freq
    trainer = pl.Trainer(
        callbacks=[checkpoint, swa], max_epochs=max_epochs, gpus=[0],
        default_root_dir=str(output_dir), stochastic_weight_avg=True,
        fast_dev_run=args.fast_dev_run,
        precision=16 if args.use_amp else 32,
        # 1 epochあたりのバッチ数を制限して、epoch終了時のSWAの処理を実行させる
        limit_train_batches=args.swa_freq,
        check_val_every_n_epoch=interval
    )
    trainer.fit(model, train_dataloader=train_loader,
                val_dataloaders=val_loader)
    metrics = trainer.test(model, test_dataloaders=test_loader)
    if isinstance(metrics, list):
        metrics = metrics[-1]
    with (output_dir / 'result.txt').open('w') as f:
        for key, value in metrics.items():
            f.write('{}: {}\n'.format(key, value))


if __name__ == '__main__':
    main()
