# Swin Transformer

## Architecture

### Stage 1

I have an input image with shape `(H, W, 3)`. The image is splitted into multiple patches. Each
patch has shape `(4, 4, 3)`. The patch is equivalent to a token in NLP. In this stage, there are
`H / 4 * W / 4` tokens.

