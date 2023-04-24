import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import io
from torchvision.transforms import ToTensor
from PIL import Image

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Classifier(nn.Module):
    def __init__(self, dlvm, num_samples=0, *args, **kwargs) -> None:
        super(Classifier, self).__init__(*args, **kwargs)
        self.dlvm = dlvm
        self.net = CNNModel()
        self.num_samples = num_samples
        if num_samples < 0:
            raise ValueError("Number of samples cannot be negative")
    
    def forward(self, inputs):
        if self.training:
            outs = []
            images = []
            images.append(inputs)
            outs.append(self.net(inputs))
            if self.num_samples > 0:
                distrib = self.dlvm.encode(inputs)
                z = distrib.rsample((self.num_samples,))
                for sample in z:
                    image = self.dlvm.decode(sample)
                    outs.append(self.net(image))
                    images.append(image)
            images = torch.cat(images, dim=0)
            outs = torch.cat(outs, dim=0)
            return outs, images
        else:
            return self.net(inputs), None

def calc_accuracy(y, y_pred):
    pred = y_pred.data.max(1, keepdim=True)[1]
    return pred.eq(y.data.view_as(pred)).sum().item()

def cal_loss_latent_space(criterion, preds, targets, num_samples):
    targets = torch.cat([targets for _ in range(num_samples + 1)], dim=0)
    return criterion(preds, targets), targets

def train_step(model, criterion, optimizer, dataloader, num_samples, epoch, writer=None):
    model.train()
    running_loss = 0
    running_acc = 0
    N = 0
    device = next(model.parameters()).device
    for x, y in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output, latent_img = model(x)
        loss, labels = cal_loss_latent_space(criterion, output, y, num_samples)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        accuracy = calc_accuracy(labels, output)
        running_acc += accuracy
        N += x.size(0)

    running_loss /= N
    running_acc /= N
    if writer is not None:
        plot_buf = plot_image(x[-8:], output[-8:], y[-8:], latent_img[-8:], labels[-8:])
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
            output, _ = model(x)
            loss = criterion(output, y)
            running_loss += loss.item()
            accuracy = calc_accuracy(y, output)
            running_acc += accuracy
            N += x.size(0)

    running_loss /= N
    running_acc /= N
    if writer is not None:
        plot_buf = plot_image(x[-8:], output[-8:], y[-8:])
        image = Image.open(plot_buf)
        image = ToTensor()(image).unsqueeze(0)
        writer.add_scalar("loss/val", running_loss, epoch)
        writer.add_scalar("accuracy/val", running_acc, epoch)
        writer.add_images("samples/val", image, epoch)
    return {"loss":running_loss, "accuracy":running_acc}

def plot_image(images, preds, targets, latent_img=None, latent_label=None):
    images = torch.permute(images, (0, 2, 3, 1))
    preds = preds.data.max(1)[1]
    cmap = None
    if images.shape[3] < 3:
        images = torch.squeeze(images)
        cmap = "gray"
    images, preds, targets = (images.cpu().detach().numpy(), 
                            preds.cpu().detach().numpy(), 
                            targets.cpu().detach().numpy())
    if latent_label is not None and latent_img is not None:
        latent_img = torch.permute(latent_img, (0, 2, 3, 1))
        latent_img = latent_img.cpu().detach().numpy()
        latent_label = latent_label.cpu().detach().numpy()
        fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(15, 30))
        fig.tight_layout()
        axes = axes.ravel()
        i = 0
        for img, pred, label, latent, llabel in zip(images, preds, targets, latent_img, latent_label):
            axes[i].imshow(img, cmap=cmap)
            axes[i].set_title(f"Label: {label} | Pred: {pred}")
            axes[i+1].imshow(latent, cmap=cmap)
            axes[i+1].set_title(f"Label: {llabel} | Pred: {pred}")
            axes[i].grid(False)
            axes[i].axis('off')
            axes[i+1].grid(False)
            axes[i+1].axis('off')
            i += 2
    else:
        fig, axes = plt.subplots(nrows=4, ncols=2)
        fig.tight_layout()
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
