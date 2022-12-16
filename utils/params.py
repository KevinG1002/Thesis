import sys
import argparse


sys.path.append("../")


def argument_parser():
    parser = argparse.ArgumentParser(description="Thesis experiment paramerter parser.")
    parser.add_argument("-d", "--dataset", default="../datasets/MNIST", type=str)
    parser.add_argument("-e", "--num_epochs", default=50, type=int)
    parser.add_argument("-b", "--batch_size", default=32, type=int)
    parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0.999, type=float)
    params = parser.parse_args()

    return params
