import sys

import click
import glob
import librosa
from more_itertools import chunked
import numpy as np
import openl3
# import resampy
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm

# Get YamNet code and pre-trained model weights in the same folder yamnet_root/
# https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
sys.path.append('.../yamnet_root/')
import params as yamnet_params
import yamnet as yamnet_model


def extract_openl3(X, batch_size=64):
    # OpenL3 takes sr=48000
    sr = 48000
    model = openl3.models.load_audio_embedding_model(input_repr='mel256', content_type='env', embedding_size=512)
    embeddings = []
    for chunk in tqdm(chunked(list(X), batch_size)):
        embs, _ = openl3.get_audio_embedding(chunk, [sr] * len(chunk), model=model, content_type='env', input_repr='mel256', embedding_size=512, batch_size=batch_size)
        embeddings.extend(embs)
    return np.array(embeddings)


def extract_yamnet(X):
    # YamNet takes sr=16000
    sr = 16000
    params = yamnet_params.Params(sample_rate=sr, patch_hop_seconds=0.1)
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('.../yamnet_root/yamnet.h5')
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
@click.option('--algo', type=click.Choice(['openl3', 'yamnet', 'resnet', 'vgg'], required=True, help='Embedding algorithms.'))
@click.option('--output_dir', required=True, help='Output folder.')
def extract_embedding(input_dir, algo, output_dir):
    files = glob.glob(input_dir)
    if files[0].endswith('wav'):
        if algo == 'openl3':
            sr = 48000
            extractor = extract_openl3
        elif algo == 'yamnet':
            sr = 16000
            extractor = extract_yamnet
        else:
            assert False
        for f in files:
            X.extend(librosa.load(f, sr=sr))
        embeddings = extractor(X)
    elif files[0].endswith('mp4'):
        if algo == 'resnet':
            extractor = extract_resnet
        elif algo == 'vgg':
            extractor = extract_vgg
        else:
            assert False
        # TODO: Read video and sample images
        embeddings = extractor(X)
    else:
        assert False
    X_train, X_valid = train_test_split(embeddings, test_size=0.1)
    np.save('{}/{}_train.npy'.format(output_dir, algo), X_train)
    np.save('{}/{}_valid.npy'.format(output_dir, algo), X_valid)


if __name__ == '__main__':
    extract_embedding()
