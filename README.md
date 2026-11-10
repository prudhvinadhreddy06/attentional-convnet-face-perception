# Attentional ConvNet & Deep Transfer Learning for Visual Perception

A PyTorch computer vision implementation comparing custom convolutional architectures with spatial attention mechanisms against transfer-learned deep residual networks (ResNet-18) for fine-grained facial perception and age estimation.

## Overview

Accurate feature extraction in visual perception tasks requires models to focus on key salient regions (such as facial landmarks). This repository implements:

1. **Custom Attentional ConvNet (`AttentionalConvNet`)**:
   - Custom 2D convolutional feature extraction layers (`nn.Conv2d` with strided convolutions).
   - Spatial soft-attention module generating dynamic sigmoid attention masks to weight salient convolutional activations.
   - Fully connected projection head for continuous regression.
2. **Transfer Learning with Deep Residual Networks (`AgePredictionModel`)**:
   - Fine-tuned **ResNet-18** backbone pre-trained on ImageNet.
   - Modified fully connected classification/regression head.
   - Customized optimization and learning rate scheduling in PyTorch.

## Architecture Highlights

```python
class AttentionalConvNet(nn.Module):
    def __init__(self, num_age_groups):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        
        # Spatial attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_age_groups)
```

## Dataset & Preprocessing Pipeline

- **Torchvision Compose Transforms**:
  - Image resizing to 224x224.
  - RGB normalization with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- Custom `torch.utils.data.Dataset` and multi-threaded `DataLoader` with batching and shuffling.

## Getting Started

```bash
# Clone the repository
git clone https://github.com/prudhvinadhreddy06/attentional-convnet-face-perception.git
cd attentional-convnet-face-perception

# Run training
python train_attentional_cnn.py
```
