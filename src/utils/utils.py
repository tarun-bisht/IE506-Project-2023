import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from sklearn.decomposition import PCA


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

def plot_latent_space(model, num_images=30, figsize=15, save_path=None):
    n_cols = num_images//2
    n_rows = num_images - n_cols
    _, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(figsize, figsize))
    images = model.sample(num_points=num_images)
    axes = axes.ravel()
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].grid(False)
        axes[i].axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_label_clusters(model, data, labels):
    compressed = model.encode(data)
    if len(compressed) > 1:
        compressed = compressed[0]
    if compressed.shape[1] > 3:
        print("Applying PCA")
        pca = PCA(n_components=2)
        compressed = pca.fit_transform(compressed)
    else:
        plt.figure(figsize=(12, 10))
        plt.scatter(compressed[:, 0], compressed[:, 1], c=labels)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
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
