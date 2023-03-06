import torch
import pyro
import seaborn as sns
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from models.mlp import MLP
from datasets.pgm_dataset import PGMDataset
from datasets.get_dataset import DatasetRetriever
from frameworks.mlp_pgm import MLP_PGM
from frameworks.sgd_template import SupervisedLearning
import datetime
from utils.exp_logging import Logger
from utils.graphical_model_utils import sample_dict_to_module
import os

EXPERIMENTAL_RESULTS_PATH = "experimental_results"


class CONFIG:
    def __init__(
        self,
        learning_rate: float,
        batch_size: int,
        num_iterations: int,
        base_model: nn.Module,
        weight_dataset: list[torch.Tensor],
        bias_dataset: list[torch.Tensor],
        num_latent_samples: int,
        num_weight_samples: int,
        log_training: bool = False,
    ) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.base_model = base_model
        self.weight_dataset = weight_dataset
        self.bias_dataset = bias_dataset
        self.num_latent_samples = num_latent_samples
        self.num_weight_samples = num_weight_samples
        self.log_training = log_training

        self.base_model_name = self.base_model.name
        self.dataset_len = self.bias_dataset[0].size()[0]
        self.experiment_config = {
            k: v
            for k, v in self.__dict__.items()
            if type(v) in [str, int, float, bool, tuple, list[int]]
        }
        print("\nExperiment Config:\n%s" % self.experiment_config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.experiment_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_SVI_MLP_PGM_{num_iterations}_its_{batch_size}_b_{learning_rate}_lr"
        if log_training:
            if os.path.isdir(EXPERIMENTAL_RESULTS_PATH):
                self.experiment_dir = os.path.join(
                    EXPERIMENTAL_RESULTS_PATH, self.experiment_name
                )
                if not os.path.exists(self.experiment_dir):
                    os.mkdir(self.experiment_dir)

                self.logger = Logger(
                    self.experiment_name, self.experiment_dir, self.experiment_config
                )
        else:
            self.experiment_dir = None

        assert self.batch_size <= self.dataset_len, "Batch size larger than dataset"


def run_experiment(cfg: CONFIG):
    print(cfg.device)
    pgm_process = MLP_PGM(
        cfg.learning_rate,
        cfg.batch_size,
        cfg.num_iterations,
        cfg.base_model,
        cfg.dataset_len,
        cfg.device,
    )
    mnist_examples = DatasetRetriever("MNIST", False)
    train_set, _ = mnist_examples()
    sample_idxs = torch.randint(0, len(train_set), (cfg.dataset_len,))
    # example_samples = train_set.data[sample_idxs].flatten(1).unsqueeze(-1)
    subsamples = Subset(train_set, sample_idxs)
    dataloader = DataLoader(subsamples, batch_size=len(subsamples))
    example_samples = next(iter(dataloader))[0].flatten(1).unsqueeze(-1)
    # print(example_samples.size())
    # print(example_samples[0])

    pgm_process.svi_train(
        example_samples, cfg.weight_dataset, cfg.bias_dataset)
    latents = pgm_process.sample_latents(cfg.num_latent_samples)
    weight_samples = pgm_process.sample_weights(
        example_samples, cfg.num_weight_samples)
    weight_dicts = [
        {k: v[i] for k, v in weight_samples.items()}
        for i in range(cfg.num_weight_samples)
    ]
    # nn_1 = sample_dict_to_module(pgm_process.base_model, nn_1)
    nn_s = [
        sample_dict_to_module(pgm_process.base_model, w_dict) for w_dict in weight_dicts
    ]
    gen_model_test_dataset = DatasetRetriever("MNIST")
    _, test_set = gen_model_test_dataset()

    # print("\nTesting Generated Sample 1:\n")
    sample_weight_accs = []
    for i in range(cfg.num_weight_samples):
        test_process = SupervisedLearning(
            nn_s[i], test_set=test_set, device=cfg.device)
        test_metrics = test_process.test()
        sample_weight_accs.append(test_metrics["test_acc"])

    print("Ensemble Accuracy:", sum(sample_weight_accs) / cfg.num_weight_samples)


if __name__ == "__main__":
    dataset_json_path = "/scratch_net/bmicdl03/kgolan/Thesis/datasets/model_dataset_MNIST/model_dataset.json"
    dataset_obj = PGMDataset(
        model_json_path=dataset_json_path, base_model=MLP())
    weights, biases = dataset_obj()
    # print(weights[0].size(), weights[0])

    cfg = CONFIG(
        learning_rate=2e-3,
        batch_size=100,
        num_iterations=10000,
        base_model=MLP(),
        weight_dataset=weights,
        bias_dataset=biases,
        num_latent_samples=40,
        num_weight_samples=40,
        log_training=False,
    )
    run_experiment(cfg)
