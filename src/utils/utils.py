import matplotlib.pyplot as plt
import torch
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def plot_dataloader(dataloader, labelmap, nrows=2, ncols=4):
    imgs, labels = next(iter(dataloader))
    imgs = torch.permute(imgs, (0, 2, 3, 1))
    if imgs.shape[3] < 3:
        imgs = torch.squeeze(imgs)
    _, axes = plt.subplots(nrows=nrows, ncols=ncols)
    imgs, labels = imgs.numpy(), labels.numpy()
    axes = axes.ravel()
    for i, (img, label) in enumerate(zip(imgs, labels)):
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {labelmap[label]}")
        axes[i].grid(False)
        axes[i].axis('off')
    plt.show()
        

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '../..')

    from src.data.datasets import mnist_dataset, cifar10_dataset, fashion_dataset
    mnist_loader, mnist_labelmap = mnist_dataset(batch_size=8, dataset_path="../../temp")
    cifar_loader, cifar_labelmap = cifar10_dataset(batch_size=8, dataset_path="../../temp")
    fashion_loader, fashion_labelmap = fashion_dataset(batch_size=8, dataset_path="../../temp")
    plot_dataloader(mnist_loader, mnist_labelmap)
    plot_dataloader(cifar_loader, cifar_labelmap)
    plot_dataloader(fashion_loader, fashion_labelmap)
