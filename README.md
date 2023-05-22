# PredatorX-Text-to-Image-Model

Conditional Generative Adversarial Network (CGAN) for Image and Text Fusion
This repository contains the implementation of a Conditional Generative Adversarial Network (CGAN) for fusing image and text data. The model consists of a generator and a discriminator, both implemented using the PyTorch deep learning framework.

## Prerequisites
Before running the code, make sure you have the following dependencies installed:

- Python 3.x
- PyTorch
You can install PyTorch by following the instructions on the PyTorch website.

## Model Architecture
The CGAN consists of the following components:

### Generator
The Generator class is responsible for processing the text and image inputs and generating an output image. It consists of the following layers:

Text Processing Layers:

Embedding layer: Maps the input text to a dense vector representation.
LSTM layer: Processes the embedded text and captures its sequential information.
Image Processing Layers:

Convolutional layers: Extract features from the input image using convolution operations.
ReLU activation: Applies the rectified linear unit activation function to introduce non-linearity.
Max pooling: Reduces the spatial dimensions of the feature maps.
Fusion Layers:

Fully connected layers: Combine the text and image features.
ReLU activation: Introduces non-linearity.
Linear layer: Produces a dense vector representation of the fused features.
Tanh activation: Generates the output image using the hyperbolic tangent function.
Discriminator
The Discriminator class is responsible for discriminating between real and generated images. It takes an image as input and predicts its authenticity. It consists of the following layers:

Convolutional layers: Extract features from the input image using convolution operations.
ReLU activation: Applies the rectified linear unit activation function to introduce non-linearity.
Max pooling: Reduces the spatial dimensions of the feature maps.
Fully connected layers: Transform the image features into a single probability value.
Sigmoid activation: Produces a probability score between 0 and 1 for binary classification.
ImageTextCGAN
The ImageTextCGAN class combines the generator and discriminator into a single model. It takes both text and image inputs and generates an output image while also predicting its authenticity.

## Usage
To use the CGAN model, follow these steps:

Initialize the CGAN model by creating an instance of the ImageTextCGAN class with the desired hyperparameters.

Prepare the input data:

For the text_input, create a tensor with the shape (seq_length,) containing the indices of the words in the input text. Replace the placeholder values with actual text data.
For the image_input, create a tensor with the shape (batch_size, image_channels, image_size, image_size) containing the pixel values of the input image. Replace the placeholder values with actual image data.
Call the forward method of the model, passing the text_input and image_input tensors as arguments. This will generate an output image and predict its authenticity.

Access the generated image and validity predictions using the returned values.

Here's an example usage:
```python
import torch

# Initialize the conditional GAN model
text_embedding_dim = 128
image_channels = 3
hidden_size = 256
num_layers = 2
num_filters = 64
kernel_size = 3
stride = 2
fusion_hidden_size = 512
image_size = 24

model = ImageTextCGAN(text_embedding_dim, image_channels, hidden_size, num_layers, num_filters, kernel_size, stride, fusion_hidden_size, image_size)

# Example usage
text_input = torch.tensor([1, 2, 3])  # Placeholder input, replace with actual text data
image_input = torch.randn(1, image_channels, image_size, image_size)  # Placeholder input, replace with actual image data

generated_image, validity = model(text_input, image_input)
print(generated_image.shape)  # Output shape: (1, 1, 24, 24)
print(validity.shape)  # Output shape: (1, 1)
```

Replace the placeholder values with actual data to use the CGAN for your specific task.

### Conclusion
This repository provides an implementation of a Conditional Generative Adversarial Network (CGAN) for fusing image and text data. The model can be used for various applications, such as generating images based on text descriptions or enhancing image generation by incorporating textual context. Feel free to explore and modify the code to suit your needs!

