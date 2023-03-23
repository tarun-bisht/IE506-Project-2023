from torchvision.utils import make_grid

def val_step(model, dataloader, epoch, writer=None):
    model.eval()
    running_loss = 0
    running_rec = 0
    running_elbo = 0
    N = 0
    device = next(model.parameters()).device
    for x, _ in dataloader:
        x = x.to(device)
        output = model(x)
        loss = output["loss"]
        rec_loss = output["reconstruction_loss"]
        elbo_loss = output["elbo"]
        running_loss += loss.item()*x.size(0)
        running_rec += rec_loss.item()*x.size(0)
        running_elbo += elbo_loss.item()*x.size(0)
        N += x.size(0)

    running_loss /= N
    running_rec /= N
    running_elbo /= N
    if writer is not None:
        writer.add_scalar("loss/val", running_loss, epoch)
        writer.add_scalar("reconstruction_loss/val", running_rec, epoch)
        writer.add_images("ground_truth/val", x[-8:], epoch)
        writer.add_images(f"reconstruction/val", output["output"][-8:], epoch)
        sample = model.sample(64)
        writer.add_image("sample", make_grid(sample, nrow=8), epoch)
    return {"loss":running_loss, "reconstruction_loss":running_rec, "elbo":running_elbo}
            

