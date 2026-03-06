import argparse

from datasets import SHD_dataloaders, SSC_dataloaders, GSC_dataloaders
from config import Config
from snn_delays import SnnDelays
import torch
from snn import SNN
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sigma_drop",
        type=float,
        default=None,
        help="Stddev of Gaussian noise added to delay positions during training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for this run.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional wandb/local run name prefix.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["shd", "ssc", "gsc"],
        default=None,
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--sparsity_p",
        type=float,
        default=None,
        help="Mask probability for sparse connectivity (0.0 = fully connected).",
    )
    args, _ = parser.parse_known_args()
    return args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n=====> Device = {device} \n\n")

args = parse_args()
config = Config()
if args.sigma_drop is not None:
    if args.sigma_drop < 0:
        raise ValueError("--sigma_drop must be >= 0.0")
    config.sigma_drop = args.sigma_drop
if args.dataset is not None:
    config.dataset = args.dataset
    config.n_outputs = 20 if config.dataset == "shd" else 35
if args.sparsity_p is not None:
    if not (0.0 <= args.sparsity_p <= 1.0):
        raise ValueError("--sparsity_p must be in [0.0, 1.0]")
    config.sparsity_p = args.sparsity_p
if args.seed is not None:
    config.seed = args.seed
if args.run_name is not None:
    config.run_name = args.run_name
if hasattr(config, "refresh_run_metadata"):
    config.refresh_run_metadata()

if config.model_type == 'snn':
    model = SNN(config).to(device)
else:
    model = SnnDelays(config).to(device)

if config.model_type == 'snn_delays_lr0':
    model.round_pos()


print(f"===> Dataset    = {config.dataset}")
print(f"===> Model type = {config.model_type}")
print(f"===> Seed       = {config.seed}")
print(f"===> Sigma drop = {config.sigma_drop}")
print(f"===> Sparsity p = {config.sparsity_p}")
print(f"===> Model size = {utils.count_parameters(model)}\n\n")


if config.dataset == 'shd':
    train_loader, valid_loader = SHD_dataloaders(config)
    test_loader = None
elif config.dataset == 'ssc':
    train_loader, valid_loader, test_loader = SSC_dataloaders(config)
elif config.dataset == 'gsc':
    train_loader, valid_loader, test_loader = GSC_dataloaders(config)
else:
    raise Exception(f'dataset {config.dataset} not implemented')


model.train_model(train_loader, valid_loader, test_loader, device)
