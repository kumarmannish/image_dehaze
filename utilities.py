import torch
from torchvision.utils import save_image
import configuration


def save_some_examples(generator, val_loader, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(configuration.DEVICE), y.to(configuration.DEVICE)
    generator.eval()
    with torch.no_grad():
        y_fake = generator(x)
        y_fake = y_fake * 0.5 + 0.5
        save_image(y_fake, folder + f"/y_gen_.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_.png")
    generator.train()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving Checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading Checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=configuration.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
