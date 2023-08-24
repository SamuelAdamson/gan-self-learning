import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 3e-4
Z_DIMENSION = 64
IMAGE_DIMENSION = 784 # 28 by 28 by 1
BATCH_SIZE = 32
NUM_EPOCHS = 10