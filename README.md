# crossmodal

```
> script/setup # Clone AudioSet models and download pre-trained weights
> python -m crossmodal.embedding --input_dir '.../audio/' --algo 'openl3' --output_dir '.../embeddings/'
> python crossmodal.embedding --input_dir '.../video/' --algo 'resnet' --output_dir '.../embeddings/'
> python crossmodal.train --embedding_dir '.../embeddings/' --model_dir '.../models/' --audio_alg 'openl3' --image_alg 'resnet'
```
