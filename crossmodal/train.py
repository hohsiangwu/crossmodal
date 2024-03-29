
from argparse import Namespace
import logging
import os
import pickle
import warnings
logging.disable(logging.INFO)
warnings.filterwarnings('ignore')

from omegaconf import DictConfig
import hydra
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.base import LightningLoggerBase
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset, IterableDataset

from .model import Baseline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DictLogger(LightningLoggerBase):

    def __init__(self, version):
        super(DictLogger, self).__init__()
        self.metrics = []
        self.params = []
        self._version = version

    @property
    def experiment(self):
        return ''

    @property
    def name(self):
        return ''

    def log_hyperparams(self, params):
        self.params.append(params)

    def log_metrics(self, metrics, step=None):
        self.metrics.append(metrics)

    @property
    def version(self):
        return self._version


dim_map = {
    'resnet': 2048,
    'vgg': 512,
    'yamnet': 1024,
    'openl3': 512,
}


@hydra.main(config_path='conf', config_name='config')
def train(cfg: DictConfig) -> None:
    hparams = {
        'audio_alg': cfg.train.audio_alg,
        'image_alg': cfg.train.image_alg,
        'batch_size': cfg.train.batch_size,
        'input1_dim': dim_map[cfg.train.audio_alg],
        'input2_dim': dim_map[cfg.train.image_alg],
        'middle_dim': 256,
        'output_dim': 128,
        'lr': cfg.train.lr,
        'num_workers': cfg.train.num_workers,
        'num_gpus': cfg.train.num_gpus,
    }
    hparams = Namespace(**hparams)

    audio_embeddings = np.load('{}/{}.npy'.format(cfg.train.embedding_dir, hparams.audio_alg))
    image_embeddings = np.load('{}/{}.npy'.format(cfg.train.embedding_dir, hparams.image_alg))
    a_train, a_valid, i_train, i_valid = train_test_split(audio_embeddings, image_embeddings, test_size=0.1, random_state=42)

    log_str = '{}-{}-{}-{}'.format(cfg.train.audio_alg, cfg.train.image_alg, 'mlp', cfg.train.batch_size)

    model_path = '{}/{}/models/{}-'.format(cfg.train.model_dir, log_str, log_str)
    model_path = model_path + '{epoch}-{val_loss:.4f}'

    logger = DictLogger(log_str)
    tensorboard_logger = TensorBoardLogger('{}/{}/logs/'.format(cfg.train.model_dir, log_str), name=log_str)

    baseline = Baseline(hparams)
    trainer = Trainer(logger=[logger, tensorboard_logger],
                      callbacks=[EarlyStopping(monitor='val_loss', patience=5),
                                 ModelCheckpoint(filepath=model_path, monitor='val_loss', save_top_k=-1)],
                      gpus=hparams.num_gpus,
                      accelerator='dp',
                      max_epochs=1000)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(a_train), torch.from_numpy(i_train)),
                              batch_size=hparams.batch_size,
                              shuffle=True)

    val_loader =  DataLoader(TensorDataset(torch.from_numpy(a_valid), torch.from_numpy(i_valid)),
                             batch_size=hparams.batch_size,
                             shuffle=False)
    trainer.fit(baseline, train_loader, val_loader)


if __name__ == '__main__':
    train()
