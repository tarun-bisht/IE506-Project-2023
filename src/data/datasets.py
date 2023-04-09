import torch
import torchvision.transforms as ttf
import torchvision.datasets as tds


def mnist_dataset(batch_size, dataset_path="./temp", val=False):
    labelmap = {idx: idx for idx in range(10)}
    train_dataset = tds.MNIST(root=dataset_path, train=True, download=True, transform=ttf.Compose([ttf.ToTensor()]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val:
        val_dataset = tds.MNIST(root=dataset_path, train=False, download=True, transform=ttf.Compose([ttf.ToTensor()]))
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        return train_dataloader, val_dataloader, labelmap
    return train_dataloader, labelmap

def cifar10_dataset(batch_size, dataset_path="./temp", val=False):
    labelmap = {idx: label for idx, label in enumerate(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])}
    train_dataset = tds.CIFAR10(root=dataset_path, train=True, download=True, transform=ttf.Compose([ttf.ToTensor()]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val:
        val_dataset = tds.CIFAR10(root=dataset_path, train=False, download=True, transform=ttf.Compose([ttf.ToTensor()]))
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        return train_dataloader, val_dataloader, labelmap
    return train_dataloader, labelmap

def fashion_dataset(batch_size, dataset_path="./temp", val=False):
    labelmap = {idx: label for idx, label in enumerate(["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"])}
    train_dataset = tds.FashionMNIST(root=dataset_path, train=True, download=True, transform=ttf.Compose([ttf.ToTensor()]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val:
        val_dataset = tds.FashionMNIST(root=dataset_path, train=False, download=True, transform=ttf.Compose([ttf.ToTensor()]))
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        return train_dataloader, val_dataloader, labelmap
    return train_dataloader, labelmap

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '../..')

    print("Train Data")
    mnist_loader, mnist_labelmap = mnist_dataset(batch_size=8, dataset_path="../../temp")
    cifar_loader, cifar_labelmap = cifar10_dataset(batch_size=8, dataset_path="../../temp")
    fashion_loader, fashion_labelmap = fashion_dataset(batch_size=8, dataset_path="../../temp")
    print(next(iter(mnist_loader))[0][0].shape)
    print(next(iter(cifar_loader))[0][0].shape)
    print(next(iter(fashion_loader))[0][0].shape)
    print("Val Data")
    mnist_loader, mnist_val, mnist_labelmap = mnist_dataset(batch_size=8, dataset_path="../../temp", val=True)
    cifar_loader, cifar_val, cifar_labelmap = cifar10_dataset(batch_size=8, dataset_path="../../temp", val=True)
    fashion_loader, fashion_val, fashion_labelmap = fashion_dataset(batch_size=8, dataset_path="../../temp", val=True)
    print("Val data")
    print(next(iter(mnist_val))[0][0].shape)
    print(next(iter(cifar_val))[0][0].shape)
    print(next(iter(fashion_val))[0][0].shape)

    print("Min Max Values")
    print(next(iter(mnist_val))[0][0].max())
    print(next(iter(mnist_val))[0][0].min())
    print(next(iter(cifar_val))[0][0].max())
    print(next(iter(cifar_val))[0][0].min())
    print(next(iter(fashion_val))[0][0].max())
    print(next(iter(fashion_val))[0][0].min())