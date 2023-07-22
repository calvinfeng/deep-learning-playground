# Deep Learning Playground

## TODO

- [ ] Parallel batch encoding for all models
- [ ] Data augmentation logic
- [x] Tensorboard Visualizer
- [x] Implement NMS for decoding
- [x] SSD
  - [x] Anchor Generator
  - [x] Encoder
  - [x] Decoder
  - [x] Loss
  - [x] Training Loop
- [x] CenterNet
  - [x] Keypoint Encoder
  - [ ] Keypoint Decoder
  - [ ] Loss
  - [ ] Training Loop
- [ ] Swin Transformer
- [ ] ResNeXT

## Parallel Encoding Example

```py
import torch.multiprocessing as mp

def encode_sample(sample):
    # This function encodes a single sample.
    # Add your own encoding logic here.
    return encoded_sample

def encode_batch(batch):
    with mp.Pool() as pool:
        encoded_batch = pool.map(encode_sample, batch)
    return torch.stack(encoded_batch)
```

## PyTorch Details

- Use `view` (in PyTorch) or `reshape` (in numpy) to change the shape of the data without changing
  the order of the axes. `view` requires that Tensor is contiguous.
- Use `permute` (in PyTorch) or `transpose`/`swapaxes` (in numpy) to rearrange the axes of your
  data.
