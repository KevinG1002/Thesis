import torch
import copy
import json
import os
from torch.nn import CrossEntropyLoss, MSELoss
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.optim import Adam
from models.mlp import MLP, SmallMLP, RegressMLP
from frameworks.sgd_template import SupervisedLearning
from frameworks.regression_template import RegressionTemplate
from utils.params import argument_parser
from datasets.get_dataset import DatasetRetriever


class GENCONFIG:
    def __init__(
        self,
        num_runs: int = 300,
        target_dataset_name: str = None,
        target_dataset_transforms: transforms = transforms.ToTensor(),
        target_dir: str = None,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 10,
    ):
        assert target_dir, "Target directory for trained models needs to be provided."
        assert (
            target_dataset_name
        ), "Target dataset path needs to be provided to train collection of models for dataset."
        self.num_runs = num_runs
        self.target_dataset_name = target_dataset_name
        self.target_dir = target_dir
        self.target_dataset_transforms = target_dataset_transforms
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        d = DatasetRetriever(self.target_dataset_name)
        self.train_set, self.test_set = d()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


def run(cfg: GENCONFIG):
    if not os.path.exists(os.path.join(cfg.target_dir, "models/")):
        os.mkdir(f"{cfg.target_dir}/models/")

    dataset_dicts = {}

    models = [SmallMLP() for _ in range(cfg.num_runs)]
    optimizers = [
        Adam(models[i].parameters(), lr=cfg.learning_rate) for i in range(cfg.num_runs)
    ]
    print(cfg.device)
    with open(f"{cfg.target_dir}/small_model_MNIST_dataset_2000.json", "w") as file:
        json.dump(dataset_dicts, file)
    for i in range(cfg.num_runs):
        # entry = {}
        model_path = f"{cfg.target_dir}models/small_mlp_2000_MNIST_{i}.pth"
        training_process = SupervisedLearning(
            model=models[i].to(cfg.device),
            train_set=cfg.train_set,
            test_set=cfg.test_set,
            val_set=None,
            num_classes=10,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            criterion=CrossEntropyLoss(),
            device=cfg.device,
        )
        # training_process = RegressionTemplate(model=models[i].to(cfg.device),
        #                                       train_set=cfg.train_set,
        #                                       test_set=cfg.test_set,
        #                                       val_set=None,
        #                                       epochs=cfg.epochs,
        #                                       batch_size=cfg.batch_size,
        #                                       criterion=MSELoss(),
        #                                       device=cfg.device,)
        training_process.optimizer = optimizers[i]
        training_process.train()
        performance_dict = training_process.test()
        
        trained_model = copy.deepcopy(training_process.model)
        torch.save(trained_model.state_dict(), model_path)
        with open(f"{cfg.target_dir}/small_model_MNIST_dataset_2000.json", "w") as file:
            # dataset_dicts[model_path] = {
            #     testing: v.cpu().item() if type(v) == torch.Tensor else v
            #     for k, v in performance_dict.items()
            # }
            dataset_dicts[model_path] = {"test_acc" : performance_dict["test_acc"]}

            json.dump(dataset_dicts, file)


def main():
    target_dataset = "MNIST"
    target_directory = (
        f"/scratch_net/bmicdl03/kgolan/Thesis/datasets/small_model_dataset_{target_dataset}_2000/"
    )
    # target_directory = f"../datasets/model_dataset_{target_dataset}/"
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)

    training_params = argument_parser()
    learning_rate = training_params.learning_rate
    epochs = training_params.num_epochs
    batch_size = training_params.batch_size
    num_runs = training_params.n_runs

    cfg = GENCONFIG(
        target_dir=target_directory,
        target_dataset_transforms=transforms.ToTensor(),
        target_dataset_name="MNIST",
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        num_runs=num_runs,
    )
    run(cfg)


if __name__ == "__main__":
    main()
