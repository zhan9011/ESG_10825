# Model Weights

This directory is reserved for model checkpoint files such as `.pt`, `.pth`, or
`.ckpt`.

The current reproducible workflow stores generated probability and embedding
artifacts under `experiments/` because those files are inference caches, not
trainable model checkpoint weights.

Training-time class weights, task weights, sample weights, and blend weights are
configuration values managed in `config/default.yaml`, `config/train.yaml`, and
`config/inference.yaml`.
