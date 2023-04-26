import os
import argparse
from src.data.datasets import (mnist_dataset, fashion_dataset, cifar10_dataset)
import torch
import torch.nn as nn
import torch.optim as opt
from project.energy import MSEEnergy
from project.models import ConvLAE, ConvVAE, LinearLAE, LinearVAE
from project.classifier import Classifier, train_step, val_step
from torch.utils.tensorboard import SummaryWriter
from src.utils import set_seed

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
    criterion = None
    if args.loss == "MSE":
        criterion = nn.MSELoss()
    elif args.loss == "BCE":
        criterion = nn.BCELoss()
    elif args.loss == "BCELogit":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "CE":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "NLL":
        criterion = nn.NLLLoss()
    else:
        raise NotImplementedError("Specified Loss not Implemented")

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

    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights, map_location=device))

    if args.freeze:
        for param in model.parameters():
            param.requires_grad = False

    classifier = Classifier(model, args.num_samples)
    # classifier = CNNModel()
    classifier.to(device=device)
    params = classifier.parameters()
    optimizer = opt.Adam(params, lr=args.lr)

    total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print("Model: ", classifier)
    print("Number of Training Parameters: ", total_params)
    print("="*20)

    if args.log_dir is None:
        if args.model.startswith("LAE"):
            log_dir = os.path.join("experiment_5", f"{args.dataset}-model-{args.model}-nz{args.nz}-lr{args.lr}-epoch{args.epoch}-seed{args.seed}-loss{args.loss}-num_samples{args.num_samples}")
        else:
            log_dir = os.path.join("experiment_5", f"{args.dataset}-model-{args.model}-nz{args.nz}-lr{args.lr}-epoch{args.epoch}-seed{args.seed}-loss{args.loss}-num_samples{args.num_samples}")
    else:
        log_dir = args.log_dir

    # Tensorboard
    writer = SummaryWriter(os.path.join('runs', log_dir))
    # make a diectory for saving models
    if args.save_model:
        os.makedirs(os.path.join(args.model_path, log_dir), exist_ok=True)

    # train and validation
    best_acc = 0
    best_loss = torch.inf
    stop_counter = 0
    for epoch in range(1, args.epoch+1):
        train_losses = train_step(classifier, criterion, optimizer, train_dataloader, args.num_samples, epoch, writer)
        val_losses = val_step(classifier, criterion, val_dataloader, epoch, writer)

        print("EPOCH:", epoch, "Train Loss: ", train_losses["loss"], "\t", "Train Accuracy: ", train_losses["accuracy"], "\t", "Validation Loss: ", val_losses["loss"], "\t", "Val Accuracy: ", val_losses["accuracy"])
        if best_acc < val_losses["accuracy"]:
            best_acc = val_losses["accuracy"]

        stop_counter += 1
        if val_losses["loss"] < best_loss:
            stop_counter = 0
            best_loss = val_losses["loss"]
            # Save model
            if args.save_model:
                torch.save(classifier.state_dict(), os.path.join(args.model_path, log_dir, f"exp5-model-{args.seed}.pt"))
        
        if stop_counter >= args.patience:
            print("Early Stopping Exiting")
            break
    
    writer.add_hparams({"model": args.model, "dataset": args.dataset,
                        "nz": args.nz,"lr": args.lr, "batch_size": args.batch_size,
                        "epoch": args.epoch, "seed": args.seed},
                        {"loss": val_losses["loss"]})
    writer.close()
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Latent Variable LAE Paper Project Experiments 2")
    parser.add_argument('--seed', type=int, help='random seed (default: 11)', default=11)
    parser.add_argument("--dataset", type=str, choices=["MNIST", "FASHION", "CIFAR10"],
                        default="MNIST", help='dataset to be used (default: MNIST)')
    parser.add_argument("--model", type=str, choices=["LAE", "VAE"],
                        default="VAE", help='model to be used (default: VAE (Linear VAE))')
    parser.add_argument("--loss", type=str, choices=["MSE", "BCE", "BCELogit", "CE", "NLL"],
                        default="NLL", help='energy loss to be used (default: CE )')
    parser.add_argument("--dlvm_out_activation", "-dlvm_out", type=str, choices=["sigmoid", "tanh"],
                        default=None, help='output layer activation of DLVM (default: None )')
    parser.add_argument("--nz", type=int, default=2,
                        help="latent dimensionality")
    parser.add_argument("--nh1", type=int, default=256,
                        help="number of feature maps (default: 256)")
    parser.add_argument("--nh2", nargs="+", type=int, default=[32, 64],
                        help="number of feature maps (default: [32, 64])")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate (default: 1e-4)")
    parser.add_argument("--step_size", type=float, default=1e-3,
                        help="step size of ALD (default: 1e-3)")
    parser.add_argument("--num_steps", type=int, default=2)
    parser.add_argument("--metropolis_hastings", "-mh", action='store_true',
                        help="Metropolis Hastings rejection step")
    parser.add_argument("--epoch", type=int, default=10,
                        help="number of epochs (default: 50)")
    parser.add_argument("--patience", type=int, default=10,
                        help="early stopping patience (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size (default: 32)")
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--detect_anomaly", action='store_true', default=True)
    parser.add_argument("--model_path", type=str, default='models')
    parser.add_argument("--save_model", action='store_true', default=True)
    parser.add_argument("--weights", type=str, default=None, help="Path to trained weight of deep latent variable model")
    parser.add_argument("--experiment", type=int, choices=[1, 2],
                        default=1, help='model trained using which experiment (default: 1 (Linear VAE))')
    parser.add_argument("--freeze", action='store_true', default=True)
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Number of samples to samples from latent variable to calculate loss")
    args = parser.parse_args()

    best_acc = run(args=args, seed=args.seed)
    print("Best Validation Accuracy: ", best_acc)

