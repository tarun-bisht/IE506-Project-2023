import torch
from tqdm.auto import tqdm

def train_step(model, optimizer, dataloader, epoch, writer=None, detect_anomaly=False):
    model.train()
    running_loss = 0
    running_rec = 0
    N = 0
    device = next(model.parameters()).device
    with torch.autograd.set_detect_anomaly(detect_anomaly):
        for x, _ in tqdm(dataloader):
            x = x.to(device)
            output = model(x)
            loss = output["loss"]
            rec_loss = output["reconstruction_loss"]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()*x.size(0)
            running_rec += rec_loss.item()*x.size(0)
            N += x.size(0)

    running_loss /= N
    running_rec /= N
    if writer is not None:
        writer.add_scalar("loss/train", running_loss, epoch)
        writer.add_scalar("reconstruction_loss/train", running_rec, epoch)
        writer.add_images("ground_truth/train", x[-8:], epoch)
        writer.add_images(f"reconstruction/train", output["output"][-8:], epoch)
    return {"loss":running_loss, "reconstruction_loss":running_rec}
            

