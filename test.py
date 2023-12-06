import torch
from torch.utils.data import DataLoader
from haze_dataset import HazeDataset
from networks.generator import Generator
import torch.optim as optimizer
from utilities import load_checkpoint
from torchvision.utils import save_image
import configuration


generator = Generator(in_channels=3, features=64).to(configuration.DEVICE)
optimizer_gen = optimizer.Adam(generator.parameters(), lr=configuration.LEARNING_RATE, betas=(0.5, 0.999))

load_checkpoint(
            configuration.CHECKPOINT_GEN, generator, optimizer_gen, configuration.LEARNING_RATE
        )

val_dataset = HazeDataset(haze_dir=configuration.VAL_DIR+r"\hazy", clean_dir=configuration.VAL_DIR+r"\clean")
val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )


generator.eval()
x, y = next(iter(val_loader))
for i, (input_image, _) in enumerate(zip(x, y)):
    input_image = input_image.to(configuration.DEVICE)
    with torch.no_grad():
        image_gen = generator(input_image.view(1, 3, 384, 384))
        image_gen = image_gen * 0.5 + 0.5
        real_image = input_image * 0.5 + 0.5
        save_image(image_gen, f"C:\\Users\\drkum\\Downloads\\Compressed\\results\\{i+1}. Generate.png")
        save_image(real_image, f"C:\\Users\\drkum\\Downloads\\Compressed\\results\\{i+1}. Real.png")
