from torchvision.utils import make_grid
from tqdm.auto import tqdm
import torch

def val_step(model, dataloader, epoch, writer=None):
    model.eval()
    running_loss = 0
    running_rec = 0
    N = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, _ in tqdm(dataloader):
            x = x.to(device)
            output = model(x)
            loss = output["loss"]
            rec_loss = output["reconstruction_loss"]
            running_loss += loss.item()
            running_rec += rec_loss.item()
            N += x.size(0)

    running_loss /= N
    running_rec /= N
    if writer is not None:
        writer.add_scalar("loss/val", running_loss, epoch)
        writer.add_scalar("reconstruction_loss/val", running_rec, epoch)
        writer.add_images("ground_truth/val", x[-8:], epoch)
        writer.add_images(f"reconstruction/val", output["output"][-8:], epoch)
        sample = model.sample(64)
        writer.add_image("sample", make_grid(sample, nrow=8, scale_each=True), epoch)
    return {"loss":running_loss, "reconstruction_loss":running_rec}
            

