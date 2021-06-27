from omegaconf import DictConfig
import hydra
import numpy as np
import torch

from .model import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(config_path='conf', config_name='config')
def translate(cfg: DictConfig) -> None:

    checkpoint = torch.load(cfg.train.model_file, map_location=device)
    mlp = MLP(checkpoint['hyper_parameters']['input1_dim'],
              checkpoint['hyper_parameters']['input2_dim'],
              checkpoint['hyper_parameters']['middle_dim'],
              checkpoint['hyper_parameters']['output_dim'])
    mlp.load_state_dict({'.'.join(k.split('.')[1:]): v for k, v in checkpoint['state_dict'].items()})
    mlp.eval()

    audio_embeddings = np.load('{}/{}.npy'.format(cfg.translate.embedding_dir, cfg.train.audio_alg))
    image_embeddings = np.load('{}/{}.npy'.format(cfg.translate.embedding_dir, cfg.train.image_alg))

    translated_audio_embeddings, translated_image_embeddings = mlp(torch.from_numpy(audio_embeddings).to(device),
                                                                   torch.from_numpy(image_embeddings).to(device))
    np.save('{}/{}.npy'.format(cfg.translate.output_dir, cfg.train.audio_alg), translated_audio_embeddings.detach().cpu().numpy())
    np.save('{}/{}.npy'.format(cfg.translate.output_dir, cfg.train.image_alg), translated_image_embeddings.detach().cpu().numpy())


if __name__ == '__main__':
    translate()
