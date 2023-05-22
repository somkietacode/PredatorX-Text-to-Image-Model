import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, text_embedding_dim, image_channels, hidden_size, num_filters, kernel_size, stride, fusion_hidden_size, image_size):
        super(Generator, self).__init__()
        
        # Text Processing Layers
        self.embedding = nn.Embedding(vocab_size, text_embedding_dim)
        self.lstm = nn.LSTM(text_embedding_dim, hidden_size, num_layers)
        
        # Image Processing Layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(image_channels, num_filters, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size, stride=stride),
            # Add more convolutional and pooling layers if needed
        )
        
        # Fusion Layers
        self.fully_connected = nn.Sequential(
            nn.Linear(hidden_size + num_filters, fusion_hidden_size),
            nn.ReLU(),
            nn.Linear(fusion_hidden_size, image_size * image_size),
            nn.Tanh()  # Output activation function changed to Tanh for image generation
        )
    
    def forward(self, text_input, image_input):
        # Text Processing
        embedded_text = self.embedding(text_input)
        _, (text_output, _) = self.lstm(embedded_text)
        text_output = text_output[-1]
        
        # Image Processing
        image_features = self.conv_layers(image_input)
        image_features = image_features.view(image_features.size(0), -1)
        
        # Fusion
        fused_features = torch.cat((text_output, image_features), dim=1)
        generated_image = self.fully_connected(fused_features)
        generated_image = generated_image.view(-1, 1, image_size, image_size)
        
        return generated_image

class Discriminator(nn.Module):
    def __init__(self, image_channels, num_filters, kernel_size, stride, image_size):
        super(Discriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(image_channels, num_filters, kernel_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size, stride=stride),
            # Add more convolutional and pooling layers if needed
        )
        
        self.fc = nn.Sequential(
            nn.Linear(num_filters * ((image_size // stride) ** 2), 1),
            nn.Sigmoid()  # Output activation function for binary classification
        )
    
    def forward(self, image_input):
        image_features = self.conv_layers(image_input)
        image_features = image_features.view(image_features.size(0), -1)
        validity = self.fc(image_features)
        return validity

class ImageTextCGAN(nn.Module):
    def __init__(self, text_embedding_dim, image_channels, hidden_size, num_layers, num_filters, kernel_size, stride, fusion_hidden_size, image_size):
        super(ImageTextCGAN, self).__init__()
        self.generator = Generator(text_embedding_dim, image_channels, hidden_size, num_filters, kernel_size, stride, fusion_hidden_size, image_size)
        self.discriminator = Discriminator(image_channels, num_filters, kernel_size, stride, image_size)
    
    def forward(self, text_input, image_input):
        generated_image = self.generator(text_input, image_input)
        validity = self.discriminator(generated_image)
        return generated_image, validity

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
