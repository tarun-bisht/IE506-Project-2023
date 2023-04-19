import os
import argparse
from src.data.datasets import (mnist_dataset, fashion_dataset, cifar10_dataset)
import torch
import torch.nn
import torch.optim as opt
from project.energy import MSEEnergy, BCEEnergy
from project.models import ConvLAE, ConvVAE
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
        energy = BCEEnergy(logit=False)
    elif args.loss == "BCELogit":
        energy = BCEEnergy(logit=True)
    else:
        energy = None
        raise NotImplementedError("Specified Energy not Implemented")

    # Model
    if args.model == "LAE":
        model = ConvLAE(input_shape, args.nh, args.nz, energy, args.num_steps, args.step_size, args.metropolis_hastings, args.out_activation)
    elif args.model == "VAE":
        model = ConvVAE(input_shape, args.nh, args.nz, energy, args.out_activation)
    else:
        model = None
        raise NotImplementedError("Model Specified not implemented")
    model.to(device=device)
    print(model)
    params = model.parameters()
    optimizer = opt.Adam(params, lr=args.lr)

    if args.log_dir is None:
        if args.model.startswith("LAE"):
            log_dir = os.path.join("experiment_2", f"{args.dataset}-model-{args.model}-nh{args.nh}-nz{args.nz}-step_size{args.step_size}-num_steps-{args.num_steps}-{'mh' if args.metropolis_hastings else 'no_mh'}-lr{args.lr}-epoch{args.epoch}-seed{args.seed}-loss{args.loss}-out_activation{args.out_activation}")
        else:
            log_dir = os.path.join("experiment_2", f"{args.dataset}-model-{args.model}-nh{args.nh}-nz{args.nz}-lr{args.lr}-epoch{args.epoch}-seed{args.seed}-loss{args.loss}-out_activation{args.out_activation}")
    else:
        log_dir = args.log_dir

    # Tensorboard
    writer = SummaryWriter(os.path.join('runs', log_dir))
    # make a diectory for saving models
    if args.save_model:
        os.makedirs(os.path.join(args.model_path, log_dir), exist_ok=True)

    # train and validation
    best_rec = torch.inf
    best_loss = torch.inf
    stop_counter = 0
    for epoch in range(1, args.epoch+1):
        train_losses = train_step(model, optimizer, train_dataloader, epoch, writer, args.detect_anomaly)
        val_losses = val_step(model, val_dataloader, epoch, writer)

        print("EPOCH:", epoch, "Train Loss: ", train_losses["loss"], "\t", "Validation Loss: ", val_losses["loss"], "\t", "Reconstruction Loss: ", val_losses["reconstruction_loss"])
        if best_rec > val_losses["reconstruction_loss"]:
            best_rec = val_losses["reconstruction_loss"]

        stop_counter += 1
        if val_losses["loss"] < best_loss:
            stop_counter = 0
            best_loss = val_losses["loss"]
            # Save model
            if args.save_model:
                torch.save(model.state_dict(), os.path.join(args.model_path, log_dir, f"exp2-model.pt"))
        
        if stop_counter >= args.patience:
            print("Early Stopping Exiting")
            break

    val_losses = val_step(model, val_dataloader, epoch, writer)
    
    writer.add_hparams({"model": args.model, "dataset": args.dataset,
                        "nz": args.nz, "nh": torch.tensor(args.nh),
                        "lr": args.lr, "batch_size": args.batch_size,
                        "epoch": args.epoch, "seed": args.seed},
                        {"loss": val_losses["loss"]})
    writer.close()
    return best_rec

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Latent Variable LAE Paper Project Experiments 2")
    parser.add_argument('--seed', type=int, help='random seed (default: 11)', default=11)
    parser.add_argument("--dataset", type=str, choices=["MNIST", "FASHION", "CIFAR10"],
                        default="MNIST", help='dataset to be used (default: MNIST)')
    parser.add_argument("--model", type=str, choices=["LAE", "VAE"],
                        default="VAE", help='model to be used (default: VAE (Linear VAE))')
    parser.add_argument("--loss", type=str, choices=["MSE", "BCE", "BCELogit"],
                        default="MSE", help='energy loss to be used (default: MSE )')
    parser.add_argument("--out_activation", type=str, choices=["sigmoid", "tanh"],
                        default=None, help='output layer activation (default: sigmoid )')
    parser.add_argument("--nz", type=int, default=2,
                        help="latent dimensionality")
    parser.add_argument("--nh", nargs="+", type=int, default=[32, 64],
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
    args = parser.parse_args()

    seeds = [11]
    recs = []
    for seed in seeds:
        rec = run(args=args, seed=seed)
        recs.append(rec)
    rec_mean = torch.mean(torch.tensor(recs)).item()
    rec_std = torch.std(torch.tensor(recs)).item()
    print("rec_mean: ", rec_mean)
    print("rec_std: ", rec_std)
    with open("log.txt", 'a') as logger:
        msg = f"rec_mean: {rec_mean}\telbo_std: {rec_std}\n \n"
        logger.write(msg)

