import torch
from utilities import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optimizer
from haze_dataset import HazeDataset
from networks.generator import Generator
from networks.discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import configuration

torch.backends.cudnn.benchmark = True


def train_fn(discriminator, generator, loader, optimizer_disc, optimizer_gen, l1_loss, bce, gen_scale, disc_scale):
    loop = tqdm(loader, leave=True)

    for index, (x, y) in enumerate(loop):
        x = x.to(configuration.DEVICE)
        y = y.to(configuration.DEVICE)

        with torch.cuda.amp.autocast():
            y_fake = generator(x)
            disc_real = discriminator(x, y)
            disc_real_loss = bce(disc_real, torch.ones_like(disc_real))
            disc_fake = discriminator(x, y_fake.detach())
            disc_fake_loss = bce(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = (disc_real_loss + disc_fake_loss)/2

        discriminator.zero_grad()
        disc_scale.scale(disc_loss).backward()
        disc_scale.step(optimizer_disc)
        disc_scale.update()

        with torch.cuda.amp.autocast():
            disc_fake = discriminator(x, y_fake)
            gen_fake_loss = bce(disc_fake, torch.ones_like(disc_fake))
            l1 = l1_loss(y_fake, y) * configuration.L1_LAMBDA
            gen_loss = gen_fake_loss + l1

        optimizer_gen.zero_grad()
        gen_scale.scale(gen_loss).backward()
        gen_scale.step(optimizer_gen)
        gen_scale.update()

        if index % 10 == 0:
            loop.set_postfix(
                disc_real = torch.sigmoid(disc_real).mean().item(),
                disc_fake = torch.sigmoid(disc_fake).mean().item(),

            )

def main():
    discriminator = Discriminator(in_channels=3).to(configuration.DEVICE)
    generator = Generator(in_channels=3, features=64).to(configuration.DEVICE)
    optimizer_disc = optimizer.Adam(discriminator.parameters(), lr=configuration.LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_gen = optimizer.Adam(generator.parameters(), lr=configuration.LEARNING_RATE, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    if configuration.LOAD_MODEL:
        load_checkpoint(
            configuration.CHECKPOINT_GEN, generator, optimizer_gen, configuration.LEARNING_RATE
        )
        load_checkpoint(
            configuration.CHECKPOINT_DISC, discriminator, optimizer_disc, configuration.LEARNING_RATE
        )

    train_dataset = HazeDataset(haze_dir=configuration.TRAIN_DIR+r"\hazy", clean_dir=configuration.TRAIN_DIR+r"\clean")
    train_loader = DataLoader(
        train_dataset,
        batch_size=configuration.BATCH_SIZE,
        shuffle=False,
        num_workers=configuration.NUM_WORKERS
    )

    gen_scale = torch.cuda.amp.GradScaler()
    disc_scale = torch.cuda.amp.GradScaler()
    val_dataset = HazeDataset(haze_dir=configuration.VAL_DIR+r"\hazy", clean_dir=configuration.VAL_DIR+r"\clean")
    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False
    )

    for epoch in range(configuration.NUM_EPOCHS):
        print(f"{epoch+1}/{configuration.NUM_EPOCHS}")
        train_fn(discriminator, generator, train_loader, optimizer_disc, optimizer_gen, l1_loss, bce, gen_scale, disc_scale)

        if configuration.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(generator, optimizer_gen, filename=configuration.CHECKPOINT_GEN)
            save_checkpoint(discriminator, optimizer_disc, filename=configuration.CHECKPOINT_DISC)

        save_some_examples(generator, val_loader, folder="evaluation")

if __name__ == "__main__":
    main()
