import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from sklearn.decomposition import PCA


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

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

def plot_latent_space(model, num_images=30, figsize=15, scale=1, save_path=None):
    device = next(model.parameters()).device
    if model.latent_dim > 2:
        raise("Please give model with latent variable of 2 dimension only")
    grid_x = np.linspace(-scale, scale, num_images)
    grid_y = np.linspace(-scale, scale, num_images)[::-1]
    hidden_zs = []
    for yi in grid_y:
        for xi in grid_x:
            hidden_zs.append([xi, yi])
    hidden_zs = torch.tensor(hidden_zs, device=device, dtype=torch.float)
    fig, axes = plt.subplots(ncols=num_images, nrows=num_images, 
                            figsize=(figsize, figsize), 
                            gridspec_kw={"wspace":0.0, "hspace":0.0})
    fig.tight_layout(pad=0)
    images = model.decode(hidden_zs)
    images = torch.nn.functional.sigmoid(images)
    images = images.permute((0, 2, 3, 1))
    images = images.cpu().detach().numpy()
    cmap = None
    if images.shape[3] < 3:
        cmap = 'gray'
    axes = axes.ravel()
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap=cmap)
        axes[i].grid(False)
        axes[i].axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_label_clusters(model, data, save_path=None):
    X, Y = data
    device = next(model.parameters()).device
    distrib = model.encode(X.to(device))
    compressed = distrib.sample()
    compressed = compressed.cpu().detach().numpy()
    if compressed.shape[1] > 3:
        print("Applying PCA")
        pca = PCA(n_components=2)
        compressed = pca.fit_transform(compressed)
    plt.figure(figsize=(12, 10))
    plt.scatter(compressed[:, 0], compressed[:, 1], c=Y)
    plt.colorbar()
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
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
