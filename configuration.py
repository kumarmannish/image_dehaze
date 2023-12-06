import torch
import albumentations as a
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = r"C:\Users\drkum\Downloads\Compressed\train"
VAL_DIR = r"C:\Users\drkum\Downloads\Compressed\test"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 384
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 200
LOAD_MODEL = True
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

both_transform = a.Compose(
    [a.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE)],
    additional_targets={"image0": "image"}
)

transform_only_input = a.Compose(
    [
        a.HorizontalFlip(p=0.5),
        a.ColorJitter(p=0.2),
        a.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2()
    ]
)

transform_only_mask = a.Compose(
    [
        a.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2()
    ]
)
