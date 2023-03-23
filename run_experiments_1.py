import os
import argparse
from src.data.datasets import (mnist_dataset, fashion_dataset, cifar10_dataset)
import torch
import torch.nn
import torch.optim as opt
from project.energy import MSEEnergy, BCEEnergy
from project.models import LinearLAE, LinearVAE
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from src.utils import train_step, val_step, set_seed

def run(args, seed):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set Seed
    set_seed(seed)

    # Dataset
    input_shape = None
    if args.dataset == "MNIST":
        input_shape = (1, 28, 28)
        train_dataloader, val_dataloader, _ = mnist_dataset(batch_size=args.batch_size, val=True)
    elif args.dataset == "CIFAR10":
        input_shape = (3, 32, 32)
        train_dataloader, val_dataloader, _ = cifar10_dataset(batch_size=args.batch_size, val=True)
    elif args.dataset == "FASHION":
        input_shape = (1, 28, 28)
        train_dataloader, val_dataloader, _ = fashion_dataset(batch_size=args.batch_size, val=True)
    else:
        train_dataloader, val_dataloader, _ = None, None, None
        raise NotImplementedError("Cannot Train No data")

    # Energy
    if args.loss == "MSE":
        energy = MSEEnergy()
    elif args.loss == "BCE":
        energy = BCEEnergy()
    else:
        energy = None
        raise NotImplementedError("Specified Energy not Implemented")

    # Model
    if args.model == "LAE1":
        model = LinearLAE(input_shape, args.nh, args.nz, energy, args.num_steps, args.step_size, args.metropolis_hastings)
    elif args.model == "VAE1":
        model = LinearVAE(input_shape, args.nh, args.nz, energy)
    else:
        model = None
        raise NotImplementedError("Model Specified not implemented")
    model.to(device=device)
    params = model.parameters()
    optimizer = opt.Adam(params, lr=args.lr)

    if args.log_dir is None:
        if args.model.startswith("LAE"):
            log_dir = f"{args.dataset}-model-{args.model}-nh{args.nh}-nz{args.nz}-step_size{args.step_size}-num_steps-{args.num_steps}-{'mh' if args.metropolis_hastings else 'no_mh'}-lr{args.lr}-epoch{args.epoch}-seed{args.seed}"
        else:
            log_dir = f"{args.dataset}-model-{args.model}-nh{args.nh}-nz{args.nz}-lr{args.lr}-epoch{args.epoch}-seed{args.seed}"
    else:
        log_dir = args.log_dir

    # Tensorboard
    writer = SummaryWriter(os.path.join('runs', log_dir))

    # make a diectory for saving models
    if args.save_model:
        os.makedirs(os.path.join(args.model_path, log_dir), exist_ok=True)

    # train and validation
    best_elbo = torch.inf
    for epoch in tqdm(range(1, args.epoch+1)):
        train_losses = train_step(model, optimizer, train_dataloader, epoch, writer, args.detect_anomaly)
        val_losses = val_step(model, val_dataloader, epoch, writer)
        # Save model
        if args.save_model:
            torch.save(model.state_dict(), os.path.join(args.model_path, log_dir, f"model-{epoch}.pt"))
            if epoch > 1:
                os.remove(os.path.join(args.model_path, log_dir, f"model-{epoch-1}.pt"))
        print("Train Loss: ", train_losses["loss"], "\t", "Validation Loss: ", val_losses["loss"])
        if best_elbo > val_losses["elbo"]:
            best_elbo = val_losses["elbo"]
    val_losses = val_step(model, val_dataloader, epoch, writer)
    
    writer.add_hparams({"model": args.model, "dataset": args.dataset,
                        "nz": args.nz, "nh": args.nh,
                        "lr": args.lr, "batch_size": args.batch_size,
                        "epoch": args.epoch, "seed": args.seed},
                        {"loss": val_losses["loss"]})
    writer.close()
    return best_elbo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Latent Variable LAE Paper Project Experiment")
    parser.add_argument('--seed', type=int, help='random seed (default: 11)', default=11)
    parser.add_argument("--dataset", type=str, choices=["MNIST", "FASHION", "CIFAR10"],
                        default="MNIST", help='dataset to be used (default: MNIST)')
    parser.add_argument("--model", type=str, choices=["LAE1", "VAE1"],
                        default="VAE1", help='model to be used (default: VAE1 (Linear VAE))')
    parser.add_argument("--loss", type=str, choices=["MSE", "BCE"],
                        default="MSE", help='energy loss to be used (default: MSE )')
    parser.add_argument("--nz", type=int, default=8,
                        help="latent dimensionality")
    parser.add_argument("--nh", type=int, default=256,
                        help="number of feature maps (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate (default: 1e-4)")
    parser.add_argument("--step_size", type=float, default=1e-4,
                        help="step size of ALD (default: 1e-4)")
    parser.add_argument("--num_steps", type=int, default=2)
    parser.add_argument("--metropolis_hastings", "-mh", action='store_true',
                        help="Metropolis Hastings rejection step")
    parser.add_argument("--epoch", type=int, default=10,
                        help="number of epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size (default: 32)")
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--detect_anomaly", action='store_true')
    parser.add_argument("--model_path", type=str, default='models')
    parser.add_argument("--save_model", action='store_true')
    args = parser.parse_args()

    seeds = [11, 34, 77]
    elbos = []
    for seed in seeds:
        elbo = run(args=args, seed=seed)
        elbos.append(elbo)
    elbo_mean = torch.mean(torch.tensor(elbos)).item()
    elbo_std = torch.std(torch.tensor(elbos)).item()
    print("elbo_mean: ", elbo_mean)
    print("elbo_std: ", elbo_std)
    with open("log.txt", 'w') as logger:
        msg = f"elbo_mean: {elbo_mean}\nelbo_std: {elbo_std}"
        logger.write(msg)

