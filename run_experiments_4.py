import os
import argparse
import torch
from project.energy import MSEEnergy
from project.models import ConvLAE, ConvVAE, LinearLAE, LinearVAE
from src.utils import plot_latent_space

def run(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    input_shape = None
    if args.dataset == "MNIST":
        input_shape = (1, 28, 28)
    elif args.dataset == "CIFAR10":
        input_shape = (3, 32, 32)
    elif args.dataset == "FASHION":
        input_shape = (1, 28, 28)
    else:
        raise NotImplementedError("Cannot Dataset not implemented")

    # Energy
    energy = MSEEnergy()

    # Model
    model = None
    if args.experiment == 2:
        if args.model == "LAE":
            model = ConvLAE(input_shape, args.nh2, args.nz, energy, args.num_steps, args.step_size, args.metropolis_hastings, args.out_activation)
        elif args.model == "VAE":
            model = ConvVAE(input_shape, args.nh2, args.nz, energy, args.out_activation)
        else:
            raise NotImplementedError("Model Specified not implemented")
    elif args.experiment == 1:
        if args.model == "LAE":
            model = LinearLAE(input_shape, args.nh1, args.nz, energy, args.num_steps, args.step_size, args.metropolis_hastings, args.out_activation)
        elif args.model == "VAE":
            model = LinearVAE(input_shape, args.nh1, args.nz, energy, args.out_activation)
        else:
            raise NotImplementedError("Model Specified not implemented")
    model.to(device=device)
    print(model)
    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    save_path = None
    if args.image_name is not None:
        save_path = os.path.join("results", "experiment_4")
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, args.image_name)
    plot_latent_space(model, args.num_images, args.figsize, args.scale, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Latent Variable LAE Paper Project Experiments 3")
    parser.add_argument("--dataset", type=str, choices=["MNIST", "FASHION", "CIFAR10"],
                        default="MNIST", help='dataset to be used (default: MNIST)')
    parser.add_argument("--experiment", type=int, choices=[1, 2],
                        default=1, help='model trained using which experiment (default: 1 (Linear VAE))')
    parser.add_argument("--model", type=str, choices=["LAE", "VAE"],
                        default="VAE", help='model to be used (default: VAE (Linear VAE))')
    parser.add_argument("--out_activation", type=str, choices=["sigmoid", "tanh"],
                        default=None, help='output layer activation (default: sigmoid )')
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
    parser.add_argument("--num_images", type=int, default=30)
    parser.add_argument("--figsize", type=int, default=15)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--image_name", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()

    run(args=args)

