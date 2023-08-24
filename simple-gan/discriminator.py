import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_dimension):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(img_dimension, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),   # ensure output is between 0 and 1
        )

    def forward(self, x):
        return self.discriminator(x)