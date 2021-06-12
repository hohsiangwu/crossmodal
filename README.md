# crossmodal

[paper](https://arxiv.org/abs/2106.01149)

```
> script/setup # Clone AudioSet models and download pre-trained weights
> python -m crossmodal.embedding +embedding.input_dir='.../audio/' +embedding.algo='openl3' +embedding.output_dir='.../embeddings/'
> python -m crossmodal.embedding +embedding.input_dir='.../video/' +embedding.algo='resnet' +embedding.output_dir='.../embeddings/'
> python -m crossmodal.train +train.embedding_dir='.../embeddings/' +train.model_dir='.../models/' +train.audio_alg='openl3' +train.image_alg='resnet'
```
