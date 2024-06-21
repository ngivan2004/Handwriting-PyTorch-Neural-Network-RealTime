import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F


class NNPy(nn.Module):
    def __init__(self):
        # runs superclass constructor
        super().__init__()

        # creates an instance of layer that can flatten 28*28 image into one dimension as input to the neural network
        # self.flatten is an instance of nn.Flatten(class)
        self.flatten = nn.Flatten()

        # A sequence of layers
        self.layers = nn.Sequential(
            # Layer 1, 784 inputs -> 512 outputs
            nn.Linear(28*28, 512),
            # Activation function ReLU
            nn.ReLU(),
            # Layer 2, 512 inputs -> 512 outputs
            nn.Linear(512, 512),
            # Ditto, ReLU
            nn.ReLU(),
            # Final layer, 512 input -> 10 outputs (0 through 9)
            nn.Linear(512, 10),
        )

        """ self.layers = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )
 """

    def forward(self, x):
        # flatten the input (shape 28, 28) -> (shape 28*28)
        x = self.flatten(x)
        # go through the layers and produce an non-normalized output called logits (no softMax yet)
        output = self.layers(x)

        return output


# this model is a CNN
class NNPyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # First convolutional layer: 1 input channel, 32 output channels, 3x3 kernel size
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce size by half

            # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel size
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Reduce size by half again
        )

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            # First fully connected layer: 64*7*7 inputs (from the flattened conv layer output) -> 128 outputs
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout

            # Second fully connected layer: 128 inputs -> 10 outputs
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # Pass through the convolutional layers
        x = self.conv_layers(x)
        # Flatten the output from the conv layers
        x = self.flatten(x)
        # Pass through the fully connected layers
        x = self.fc_layers(x)

        return x
