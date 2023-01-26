import sys
import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="Thesis experiment parameter parser.")
    parser.add_argument("-d", "--dataset_name", default="MNIST", type=str)
    parser.add_argument("-e", "--num_epochs", default=50, type=int)
    parser.add_argument("-b", "--batch_size", default=32, type=int)
    parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float)
    parser.add_argument("--n_runs", default=50, type=int)
    parser.add_argument("--n_steps", default=1000, type= int)
    parser.add_argument("--weight_decay", default=0.999, type=float)
    parser.add_argument("--M_snapshots", default=10, type=int)
    parser.add_argument("--save_every", default=None, type=int)
    parser.add_argument("--experiment_dir_path", default=None, type=str)
    parser.add_argument("--checkpoint_file_path", default=None, type=str)
    params = parser.parse_args()

    return params
