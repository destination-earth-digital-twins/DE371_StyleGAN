# Perceptual Loss

It was introduced by Johnson et al. - Perceptual losses for real-time style transfer and super-resolution. (https://arxiv.org/pdf/1603.08155).

The [**PerceptualLoss**](mfai/pytorch/losses/perceptual.py#L28) class is a `torch.nn.Module` that allows to initialize a VGG-16 and compute directly the perceptual loss between a given input and target.

### Multi Scale :
The VGG-16 was originally designed for ImageNet dataset that contains 224x224 images. It can still be used with image dimensionned differently. But in case your tensors are high dimensional (ex:1024x1024) the VGG-16 features might not be able to catch fine-scale details. The *multi_scale* mode allows to compute the Perceptual Loss on different downscale version of your original tensors. For example, if your tensors are 1024x1024, the perceptual loss will be computed both on the original dimension and on its downscaled versions : 512x512 and 256x256.

### Channel handling case :

Because the VGG-16 was designed for RGB images, the perceptual loss can be computed differently depending on your tensors channel dimension. \
The *channel_iterative_mode* is done so that the loss is iteratively computed by replicating three times each channel (RGB like) so that it is compatible with VGG-16 original architecture. \

**(Case 1)** : Tensors with N!=3 channels \
**(Case 1.1)** : *channel_iterative_mode*=False : \
The VGG-16 architecture is adapted so that it can forward tensors with N channels. There will be a single forward of the network per tensor to compute the features. \
**(Case 1.2)** : *channel_iterative_mode*=True : \
The original VGG-16 architecture is kept. There will be N forwards of the network per tensor to compute the features.

**(Case 2)** : Tensors with N=3 channels \
**(Case 2.1)** : *channel_iterative_mode*=False : \
The original VGG-16 architecture is kept. There will be a single forward of the network per tensor to compute the features. \
**(Case 2.2)** : *channel_iterative_mode*=True : \
The original VGG-16 architecture is kept. There will be N forwards of the network per tensor to compute the features.

### Pre Trained
You can either choose to compute the Perceptual Loss with the ImageNet Pre-trained version of the VGG-16 or use a random version of it.

## Example
An example of PerceptualLoss usage :
```python
# In case the target and input are different everytime
inputs = torch.rand(25, 5, 128, 128)
targets = torch.rand(25, 5, 128, 128)

# Initialize the perceptual loss class
perceptual_loss_class = PerceptualLoss(channel_iterative_mode=True, in_channels=5)

# Computing Perceptual Loss
perceptual_loss = perceptual_loss_class(inputs, targets)

```
```python
# In case you need to compare different targets to the same input
inputs = torch.rand(25, 5, 128, 128)
perceptual_loss_class = PerceptualLoss(channel_iterative_mode=True, in_channels=5)

# The features of the inputs are computed and stored in the memory
perceptual_loss_class.compute_perceptual_features(inputs)

for _ in range():

  targets = torch.rand(25, 5, 128, 128)

  # The features of the targets are computed and compared to the input features
  perceptual_loss = perceptual_loss_class(targets)

```

# LPIPS

The [**LPIPS**](mfai/pytorch/losses/perceptual.py#L28) class is a `torch.nn.Module` that computes the Learned Perceptual Image Patch Similarity metric. It is using the aforementionned PerceptualLoss class so it contains the same modes.
