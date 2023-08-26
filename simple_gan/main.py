#!/usr/bin/env python3

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision

from src.discriminator import Discriminator
from src.generator import Generator
from src import utils


if __name__ == "__main__":
    discriminator = Discriminator(utils.IMAGE_DIMENSION).to(utils.DEVICE)
    generator = Generator(utils.Z_DIMENSION, utils.IMAGE_DIMENSION).to(utils.DEVICE)
    fixed_noise = torch.randn((utils.BATCH_SIZE, utils.Z_DIMENSION)).to(utils.DEVICE)

    loader = utils.get_data_loader()
    opt_discriminator = optim.Adam(discriminator.parameters(), lr=utils.LEARNING_RATE)
    opt_generator = optim.Adam(generator.parameters(), lr=utils.LEARNING_RATE)
    loss = nn.BCELoss()

    fake_writer = SummaryWriter("runs/GAN_MINST/fake")
    real_writer = SummaryWriter("runs/GAN_MINST/real")

    ## Train
    step = 0
    for epoch in range(utils.NUM_EPOCHS):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, 784).to(utils.DEVICE)
            batch_size = real.shape[0]

            ## Discriminator
            noise = torch.randn(utils.BATCH_SIZE, utils.Z_DIMENSION).to(utils.DEVICE)
            fake = generator(noise) # generate fake data

            D_real = discriminator(real).view(-1)
            loss_D_real = loss(D_real, torch.ones_like(D_real))

            D_fake = discriminator(fake.detach()).view(-1)
            loss_D_fake = loss(D_fake, torch.zeros_like(D_fake))

            D_loss = (loss_D_real + loss_D_fake) / 2
            discriminator.zero_grad()
            D_loss.backward()
            opt_discriminator.step()

            ## Generator
            output = discriminator(fake).view(-1)
            
            G_loss = loss(output, torch.ones_like(output))
            generator.zero_grad()
            G_loss.backward()
            opt_generator.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{utils.NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                        Discriminator Loss: {D_loss:.4f}, Generator Loss: {G_loss:.4f}"
                )

                with torch.no_grad():
                    fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    fake_img_grid = torchvision.utils.make_grid(fake, normalize=True)
                    real_img_grid = torchvision.utils.make_grid(data, normalize=True)

                    fake_writer.add_image(
                        "MNIST Fake Images", fake_img_grid, global_step=step
                    )
                    real_writer.add_image(
                        "MNIST Real Images", real_img_grid, global_step=step
                    )
                    step += 1
