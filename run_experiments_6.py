import os
import argparse
import gc
from tqdm.auto import tqdm
from src.data.datasets import (mnist_dataset, fashion_dataset, cifar10_dataset)
import torch
import torch.nn as nn
import torch.nn.functional as F
from project.energy import MSEEnergy
from project.models import ConvLAE, ConvVAE, LinearLAE, LinearVAE
from project.classifier import Classifier
from torch.utils.tensorboard import SummaryWriter
from src.utils import set_seed
from functools import partial


def fgsm(model, X, y, epsilon=0.3):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    with torch.enable_grad():
        pred, _ = model(X + delta)
        loss = nn.CrossEntropyLoss()(pred, y) # Calculate loss
    loss.backward() # Calculate gradient of loss with respect to input
    Xadv = X + epsilon * delta.grad.detach().sign()
    # NEW: ensuring valid pixel values
    Xadv = Xadv.clamp(0,1)
    return Xadv


def pgd(model, X, y, epsilon=0.3, step_size=0.01, num_steps=40):
    """ Construct FGSM adversarial examples on the examples X"""
    Xadv = X #+ 0
    for _ in range(num_steps):
        delta = torch.zeros_like(Xadv, requires_grad=True)
        with torch.enable_grad():
            pred, _ = model(Xadv + delta)
            loss = nn.CrossEntropyLoss()(pred, y) # Calculate loss
        loss.backward() # Calculate gradient of loss with respect to input
        Xadv = Xadv + step_size * delta.grad.detach().sign()
        # NEW: imperceptibility condition
        Xadv = torch.min(torch.max(Xadv, X - epsilon), X + epsilon)
        # NEW: ensuring valid pixel values
        Xadv = Xadv.clamp(0,1)
    return Xadv


def adversarial_test(source_model, target_model, attack, device, test_loader, step, writer=None):
    torch.cuda.empty_cache()
    gc.collect()
    source_model.to(device)
    target_model.to(device)
    source_model.eval()
    target_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            adv = attack(source_model, data, target)
            output, _ = target_model(adv)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct/len(test_loader.dataset)
    if writer is not None:
        writer.add_images("ground_truth/test", data[-8:], step)
        writer.add_images("adversarial/test", adv[-8:], step)
    print('\nAdversarial Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, test_accuracy


def run(args, seed):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set Seed
    set_seed(seed)

    # Dataset
    input_shape = None
    if args.dataset == "MNIST":
        input_shape = (1, 28, 28)
        train_dataloader, val_dataloader, label = mnist_dataset(batch_size=args.batch_size, val=True)
    elif args.dataset == "CIFAR10":
        input_shape = (3, 32, 32)
        train_dataloader, val_dataloader, label = cifar10_dataset(batch_size=args.batch_size, val=True)
    elif args.dataset == "FASHION":
        input_shape = (1, 28, 28)
        train_dataloader, val_dataloader, label = fashion_dataset(batch_size=args.batch_size, val=True)
    else:
        train_dataloader, val_dataloader, label = None, None, None
        raise NotImplementedError("Cannot Train No data")

    # Energy
    energy = MSEEnergy()

    # Model
    model = None
    if args.experiment == 2:
        if args.model == "LAE":
            model = ConvLAE(input_shape, args.nh2, args.nz, energy, args.num_steps, args.step_size, args.metropolis_hastings, args.dlvm_out_activation)
        elif args.model == "VAE":
            model = ConvVAE(input_shape, args.nh2, args.nz, energy, args.dlvm_out_activation)
        else:
            raise NotImplementedError("Model Specified not implemented")
    elif args.experiment == 1:
        if args.model == "LAE":
            model = LinearLAE(input_shape, args.nh1, args.nz, energy, args.num_steps, args.step_size, args.metropolis_hastings, args.dlvm_out_activation)
        elif args.model == "VAE":
            model = LinearVAE(input_shape, args.nh1, args.nz, energy, args.dlvm_out_activation)
        else:
            raise NotImplementedError("Model Specified not implemented")

    classifier = Classifier(model, args.num_samples)
    classifier.to(device=device)
    if args.weights is not None:
        classifier.load_state_dict(torch.load(args.weights, map_location=device))

    total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print("Model: ", classifier)
    print("Number of Parameters: ", total_params)
    print("="*20)

    if args.log_dir is None:
        if args.model.startswith("LAE"):
            log_dir = os.path.join("experiment_6", f"{args.dataset}-model-{args.model}-nz{args.nz}-seed{args.seed}-num_samples{args.num_samples}-epsilon{args.epsilon}-attack_steps{args.steps}")
        else:
            log_dir = os.path.join("experiment_6", f"{args.dataset}-model-{args.model}-nz{args.nz}-seed{args.seed}-num_samples{args.num_samples}-epsilon{args.epsilon}-attack_steps{args.steps}")
    else:
        log_dir = args.log_dir

    # Tensorboard
    writer = SummaryWriter(os.path.join('runs', log_dir))

    # attack
    attacks = []
    if args.attack == "pgd":
        for eps in args.epsilon:
            attacks.append(partial(pgd, epsilon=eps, num_steps=args.steps))
    elif args.attack == "fgsm":
        for eps in args.epsilon:
            attacks.append(partial(fgsm, epsilon=eps, num_steps=args.steps))
    else:
        raise NotImplementedError("specified attack not implemented")
    
    for i, (attack, eps) in enumerate(zip(attacks, args.epsilon)):
        loss, acc = adversarial_test(source_model=classifier,
                                    target_model=classifier,
                                    attack=attack,
                                    device=device,
                                    test_loader=val_dataloader,
                                    step=i, 
                                    writer=writer)
        writer.add_hparams({"model": args.model, "dataset": args.dataset,
                            "nz": args.nz, "batch_size": args.batch_size,
                            "seed": args.seed, "epsilon": eps, 
                            "attack_steps": args.steps}, {"adv_loss":loss, "adv_acc":acc})
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Latent Variable LAE Paper Project Experiments 2")
    parser.add_argument('--seed', type=int, help='random seed (default: 11)', default=11)
    parser.add_argument("--dataset", type=str, choices=["MNIST", "FASHION", "CIFAR10"],
                        default="MNIST", help='dataset to be used (default: MNIST)')
    parser.add_argument("--model", type=str, choices=["LAE", "VAE"],
                        default="VAE", help='model to be used (default: VAE (Linear VAE))')
    parser.add_argument("--dlvm_out_activation", "-dlvm_out", type=str, choices=["sigmoid", "tanh"],
                        default=None, help='output layer activation of DLVM (default: None )')
    parser.add_argument("--nz", type=int, default=2,
                        help="latent dimensionality")
    parser.add_argument("--nh1", type=int, default=256,
                        help="number of feature maps (default: 256)")
    parser.add_argument("--nh2", nargs="+", type=int, default=[32, 64],
                        help="number of feature maps (default: [32, 64])")
    parser.add_argument("--step_size", type=float, default=1e-3,
                        help="step size of ALD (default: 1e-3)")
    parser.add_argument("--num_steps", type=int, default=2)
    parser.add_argument("--metropolis_hastings", "-mh", action='store_true',
                        help="Metropolis Hastings rejection step")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size (default: 32)")
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--weights", type=str, default=None, help="Path to trained weight of deep latent variable model")
    parser.add_argument("--experiment", type=int, choices=[1, 2],
                        default=1, help='model trained using which experiment (default: 1 (Linear VAE))')
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Number of samples to samples from latent variable to calculate loss")
    parser.add_argument("--epsilon", nargs="+", type=float, default=[0.1, 0.2, 0.3],
                        help="number of feature maps (default: [0.1, 0.2, 0.3])")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--attack", type=str, choices=["pgd", "fgsm"],
                        default="pgd", help='attack to be used (default: pgd)') 
    args = parser.parse_args()

    run(args=args, seed=args.seed)

