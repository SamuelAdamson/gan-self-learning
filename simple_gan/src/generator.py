import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dimension, img_dimension):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(z_dimension, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dimension),
            nn.Tanh(),  # ensure value is between -1 and 1
        )

    def forward(self, x):
        return self.generator(x)