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
