import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import SimpsonsDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from utils import save_checkpoint, load_checkpoint
from torchvision.utils import make_grid


def main():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_DIR = "data/train"
    VAL_DIR = "data/val"
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    L1_LAMBDA = 10
    NUM_EPOCHS = 500
    LOAD_MODEL = False
    SAVE_MODEL = True
    CHECKPOINT_DISC = "checkpoints/disc.pth.tar"
    CHECKPOINT_GEN = "checkpoints/gen.pth.tar"

    disc = Discriminator().to(DEVICE)
    gen = Generator().to(DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    if LOAD_MODEL:
        load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, DEVICE)
        load_checkpoint(CHECKPOINT_DISC, disc, opt_disc, DEVICE)

    train_dataset = SimpsonsDataset(root_dir=TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_dataset = SimpsonsDataset(VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    writer = SummaryWriter("logs")

    for epoch in range(NUM_EPOCHS):
        gen.train()
        loop = tqdm(train_loader, leave=True)
        for idx, (x, y) in enumerate(loop):
            x = x.to(torch.float32).to(DEVICE)
            y = y.to(torch.float32).to(DEVICE)

            # Train Discriminator
            y_fake = gen(x) # B x 3 x 256 x 256
            D_real = disc(x, y) # B x 1 x 30 x 30
            D_real_loss = BCE(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

            disc.zero_grad()
            D_loss.backward()
            opt_disc.step()

            # Train generator
            D_fake = disc(x, y_fake)
            G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1

            opt_gen.zero_grad()
            G_loss.backward()
            opt_gen.step()

            if idx % 10 == 0:
                loop.set_postfix(
                    D_real=torch.sigmoid(D_real).mean().item(),
                    D_fake=torch.sigmoid(D_fake).mean().item(),
                )

        if SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=CHECKPOINT_DISC)

        gen.eval()
        for batch_idx, (x, y) in enumerate(val_loader):
            x, y = x.to(torch.float32).to(DEVICE), y.to(torch.float32).to(DEVICE)
            with torch.no_grad():
                y_fake = gen(x)
            img_grid_contour = make_grid(x)#, normalize=True)
            img_grid_real = make_grid(y)#, normalize=True)
            img_grid_fake = make_grid(y_fake)#, normalize=True)

            writer.add_image("Contour", img_grid_contour, global_step=batch_idx)
            writer.add_image("Real", img_grid_real, global_step=batch_idx)
            writer.add_image("Fake", img_grid_fake, global_step=batch_idx)


if __name__ == "__main__":
    main()
