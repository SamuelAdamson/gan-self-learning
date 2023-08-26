import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# HYPER PARAMETERS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 3e-4
Z_DIMENSION = 64
IMAGE_DIMENSION = 784 # 28 by 28 by 1
BATCH_SIZE = 32
NUM_EPOCHS = 10

def get_data_loader() -> DataLoader:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=True)

    return loader