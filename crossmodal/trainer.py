
import logging
import os
import pickle
import warnings
logging.disable(logging.INFO)
warnings.filterwarnings('ignore')

import click
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.base import LightningLoggerBase
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

    def log_metrics(self, metric, step=None):
        self.metrics.append(metric)

    @property
    def version(self):
        return self._version


dim_map = {
    'resnet': 2048,
    'vgg': 512,
    'yamnet': 1024,
    'openl3': 512,
}


@click.command()
@click.option('--embedding_dir', required=True, help='Input embedding folder.')
@click.option('--model_dir', required=True, help='Output model folder.')
@click.option('--audio_alg', type=click.Choice(['openl3', 'yamnet'], required=True, help='Audio embedding algorithms.')
@click.option('--image_alg', type=click.Choice(['resnet', 'vgg'], required=True, help='Image embedding algorithms.')
@click.option('--batch_size', default=2048)
@click.option('--lr', default=0.0001)
@click.option('--num_workers', default=1)
@click.option('--num_gpus', default=1)
def train(embedding_dir, model_dir, audio_alg, image_alg, batch_size, lr, num_workers, num_gpus):
    hparams = {
        'audio_alg': audio_alg,
        'image_alg': image_alg,
        'batch_size': batch_size,
        'input1_dim': dim_map[audio_alg],
        'input2_dim': dim_map[image_alg],
        'middle_dim': 256,
        'output_dim': 128,
        'lr': lr,
        'num_workers': num_workers,
        'num_gpus': num_gpus,
    }
    hparams = Namespace(**hparams)

    dataset = {
        'image': {
            'train': np.load('{}/{}_train.npy'.format(embedding_dir, hparams.image_alg)),
            'valid': np.load('{}/{}_valid.npy'.format(embedding_dir, hparams.image_alg)),
        },
        'audio': {
            'train': np.load('{}/{}_train.npy'.format(embedding_dir, hparams.audio_alg)),
            'valid': np.load('{}/{}_valid.npy'.format(embedding_dir, hparams.audio_alg)),
        }
    }

    log_str = '{}-{}-{}-{}'.format(saudio_alg, image_alg, 'mlp', batch_size)
    logger = DictLogger(i)

    model_path = '{}/{}/'.format(model_dir, log_str)
    model_path = model_path + '{epoch}-{val_loss:.4f}'

    baseline = Baseline(hparams)
    trainer = Trainer(logger=[logger],
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5),
                                   ModelCheckpoint(filepath=model_path, monitor='val_loss', save_top_k=-1)],
                        gpus=hparams.num_gpus,
                        accelerator='dp',
                        max_epochs=1000)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(dataset['audio']['train']).float().to('cpu'),
                                            torch.from_numpy(dataset['image']['train']).float().to('cpu')),
                                batch_size=hparams.batch_size,
                                shuffle=True)

    val_loader =  DataLoader(TensorDataset(torch.from_numpy(dataset['audio']['valid']).float().to('cpu'),
                                           torch.from_numpy(dataset['image']['valid']).float().to('cpu')),
                                batch_size=hparams.batch_size,
                                shuffle=False)
    trainer.fit(baseline, train_loader, val_loader)


if __name__ == '__main__':
    trainer()