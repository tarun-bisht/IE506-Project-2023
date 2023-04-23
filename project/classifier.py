import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import sklearn.metrics as metrics
import torch.distributions as td
import matplotlib.pyplot as plt
import io
from torchvision.transforms import ToTensor
from PIL import Image

class Classifier(nn.Module):
    def __init__(self, dlvm, num_classes, activation=None, num_samples=0, *args, **kwargs) -> None:
        super(Classifier, self).__init__(*args, **kwargs)
        self.dlvm = dlvm
        z = dlvm.latent_dim
        self.dense = nn.Linear(z, num_classes)
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax()
        self.num_samples = num_samples
        if num_samples < 1:
            raise ValueError("Number of samples should be atleast 1")
    
    def forward(self, inputs):
        distrib = self.dlvm.encode(inputs)
        if self.training:
            outs = []
            z = distrib.rsample((self.num_samples,))
            for sample in z:
                outs.append(self.dense(F.relu(sample)))
            return torch.cat(outs, dim=0)
        else:
            z = distrib.sample()
            z = F.relu(z)
            return self.dense(z)

def calc_accuracy(y, y_pred, num_samples):
    y = torch.cat([y for _ in range(num_samples)], dim=0)
    y = y.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    accuracy = metrics.accuracy_score(y, y_pred)
    return accuracy


def cal_loss_latent_space(criterion, preds, targets, num_samples):
    targets = torch.cat([targets for _ in range(num_samples)], dim=0)
    return criterion(preds, targets)

def train_step(model, criterion, optimizer, dataloader, num_samples, epoch, writer=None, detect_anomaly=False):
    model.train()
    running_loss = 0
    running_acc = 0
    N = 0
    device = next(model.parameters()).device
    with torch.autograd.set_detect_anomaly(detect_anomaly):
        for x, y in tqdm(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = cal_loss_latent_space(criterion, output, y, num_samples)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            output = F.softmax(output, dim=1)
            output = torch.argmax(output, dim=1)
            accuracy = calc_accuracy(y, output, num_samples)
            running_acc += accuracy
            N += x.size(0)

    running_loss /= N
    running_acc /= N
    if writer is not None:
        plot_buf = plot_image(x[-8:], output[-8:], y[-8:])
        image = Image.open(plot_buf)
        image = ToTensor()(image).unsqueeze(0)
        writer.add_scalar("loss/train", running_loss, epoch)
        writer.add_scalar("accuracy/train", running_acc, epoch)
        writer.add_images("samples/train", image, epoch)
    return {"loss":running_loss, "accuracy":running_acc}

def val_step(model, criterion, dataloader, epoch, writer=None):
    model.eval()
    running_loss = 0
    running_acc = 0
    N = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            running_loss += loss.item()
            output = F.softmax(output, dim=1)
            output = torch.argmax(output, dim=1)
            accuracy = calc_accuracy(y, output, num_samples=1)
            running_acc += accuracy.item()
            N += x.size(0)

    running_loss /= N
    running_acc /= N
    if writer is not None:
        plot_buf = plot_image(x[-8:], output[-8:], y[-8:])
        image = Image.open(plot_buf)
        image = ToTensor()(image).unsqueeze(0)
        writer.add_scalar("loss/val", running_loss, epoch)
        writer.add_scalar("accuracy/val", running_acc, epoch)
        writer.add_images("samples/train", image, epoch)
    return {"loss":running_loss, "accuracy":running_acc}

def plot_image(images, preds, targets):
    images = torch.permute(images, (0, 2, 3, 1))
    cmap = None
    if images.shape[3] < 3:
        images = torch.squeeze(images)
        cmap = "gray"
    fig, axes = plt.subplots(nrows=4, ncols=2)
    fig.tight_layout()
    images, preds, targets = (images.cpu().detach().numpy(), 
                            preds.cpu().detach().numpy(), 
                            targets.cpu().detach().numpy())
    axes = axes.ravel()
    for i, (img, pred, label) in enumerate(zip(images, preds, targets)):
        axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(f"Label: {label} | Pred: {pred}")
        axes[i].grid(False)
        axes[i].axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', bbox_inches='tight')
    buf.seek(0)
    return buf
