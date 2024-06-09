# ViT_practice
ViT (Visual Transformer) implementation practice
ViT employs the Transformer architecture, which can extract the global information of the image and the local information of the image.

ViT consists of the following components:

- 1. Patch Embedding
- 2. Transformer Encoder
- 3. Linear layer
- 4. Prediction head

In Transformer Encoder, the following components are used:

- 1. Encoder Block
- 2. Layer Normalization
- 3. Residual Connection
- 4. Outputs

Encoder Block consists of the following components:

- 1. Multi-head Attention
- 2. MLP
- 3. Residual Connection
- 4. Outputs
where, Multi-head Attention is used to extract the global information of the image and the local information of the image by using the attention map, created by the Multi-head Attention. Multi-head Attention works with Query, Key, and Value, where Query is the input, Key is used to calculate the attention score, and Value is used to generate the output.

