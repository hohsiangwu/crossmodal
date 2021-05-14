import glob
import random
import sys
import warnings
warnings.filterwarnings('ignore')

import click
import librosa
from more_itertools import chunked
import numpy as np
import openl3
import skvideo.io
import tensorflow as tf
from tqdm import tqdm

# Get YamNet code and pre-trained model weights in the same folder yamnet_root/
# https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
YAMNET_ROOT = '...'
sys.path.append(YAMNET_ROOT)
import params as yamnet_params
import yamnet as yamnet_model


def extract_openl3(X, batch_size=64):
    # OpenL3 sr=48000
    sr = 48000
    model = openl3.models.load_audio_embedding_model(input_repr='mel256', content_type='env', embedding_size=512)
    embeddings = []
    for chunk in tqdm(chunked(list(X), batch_size)):
        embs, _ = openl3.get_audio_embedding(chunk, [sr] * len(chunk), model=model, content_type='env', input_repr='mel256', embedding_size=512, batch_size=batch_size)
        embeddings.extend(embs)
    return np.array(embeddings)


def extract_yamnet(X):
    # YamNet sr=16000
    sr = 16000
    params = yamnet_params.Params(sample_rate=sr, patch_hop_seconds=0.1)
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('{}/yamnet.h5'.format(YAMNET_ROOT))
    embeddings = []
    for x in tqdm(X):
        # x = resampy.resample(x, 48000, params.sample_rate)
        _, emb, _ = yamnet(x)
        embeddings.append(emb)
    return np.array(embeddings)


def extract_resnet(X, batch_size=1024):
    model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', pooling='avg')
    embeddings = []
    for chunk in tqdm(chunked(X, batch_size)):
        embeddings.extend(model.predict(np.array([tf.keras.applications.resnet.preprocess_input(x) for x in chunk])))
    return np.array(embeddings)


def extract_vgg(X, batch_size=1024):
    model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', pooling='avg')
    embeddings = []
    for chunk in tqdm(chunked(X, batch_size)):
        embeddings.extend(model.predict(np.array([tf.keras.applications.vgg16.preprocess_input(x) for x in chunk])))
    return np.array(embeddings)


@click.command()
@click.option('--input_dir', required=True, help='Input folder.')
@click.option('--algo', type=click.Choice(['openl3', 'yamnet', 'resnet', 'vgg']), required=True, help='Embedding algorithms.')
@click.option('--output_dir', required=True, help='Output folder.')
def extract_embedding(input_dir, algo, output_dir):
    files = sorted(glob.glob('{}/*'.format(input_dir)))
    if files[0].endswith('wav'):
        if algo == 'openl3':
            sr = 48000
            extractor = extract_openl3
        elif algo == 'yamnet':
            sr = 16000
            extractor = extract_yamnet
        else:
            assert False
        X = []
        for f in tqdm(files):
            X.append(librosa.load(f, sr=sr)[0])
        embeddings = extractor(X)
    elif files[0].endswith('mp4'):
        if algo == 'resnet':
            extractor = extract_resnet
        elif algo == 'vgg':
            extractor = extract_vgg
        else:
            assert False
        X = []
        for f in tqdm(files):
            # TODO: Sampling images
            X.append(random.choice(skvideo.io.vread(f)))
        embeddings = extractor(X)
    else:
        assert False
    np.save('{}/{}.npy'.format(output_dir, algo), embeddings)


if __name__ == '__main__':
    extract_embedding()
