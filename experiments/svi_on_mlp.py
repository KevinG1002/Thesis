import torch
import pyro
import seaborn as sns
import torch.nn as nn
import copy
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix
from models.mlp import MLP, SmallMLP
from datasets.pgm_dataset import PGMDataset
from datasets.get_dataset import DatasetRetriever
from frameworks.mlp_pgm import MLP_PGM
from frameworks.sgd_template import SupervisedLearning
import datetime
from utils.exp_logging import Logger
from utils.graphical_model_utils import sample_dict_to_module
from utils.params import argument_parser
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
    ):
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.experiment_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_SVI_MLP_PGM_{num_iterations}_its_{batch_size}_b_{learning_rate}_lr"
        self.experiment_config = {
            k: v
            for k, v in self.__dict__.items()
            if type(v) in [str, int, float, bool, tuple, list[int]]
        }
        print("\nExperiment Config:\n%s" % self.experiment_config)

        if self.log_training:
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
                os.mkdir(EXPERIMENTAL_RESULTS_PATH)
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
    dataloader = DataLoader(subsamples, batch_size=cfg.batch_size)
    example_samples = next(iter(dataloader))[0].flatten(1).unsqueeze(-1)
    print(example_samples.size())

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
    sample_weight_metrics = {"test_loss": [],
                             "test_acc": [],
                             "f1_metric": [],
                             "recall": [],
                             "precision": [],
                             "distinct_count": []}
    for i in range(cfg.num_weight_samples):
        test_process = SupervisedLearning(
            nn_s[i], test_set=test_set, device=cfg.device)
        test_metrics = test_process.test()
        for key in sample_weight_metrics.keys():
            sample_weight_metrics[key].append(test_metrics[key])

    expanded_metrics = copy.deepcopy(sample_weight_metrics)
    for key in sample_weight_metrics.keys():
        if key != "distinct_count":
            expanded_metrics[f"mean_{key}"] = sum(
                sample_weight_metrics[key]) / cfg.num_weight_samples
            if key == "test_loss":
                expanded_metrics[f"best_{key}"] = min(
                    sample_weight_metrics[key])
            else:
                expanded_metrics[f"best_{key}"] = max(
                    sample_weight_metrics[key])

    ensemble_metrics = test_ensemble(nn_s, config=cfg)

    expanded_metrics.update(ensemble_metrics)
    print(expanded_metrics)
    # print("Average Accuracy:", sum(
    #     sample_weight_metrics["test_acc"]) / cfg.num_weight_samples)
    # print("Average Loss:", sum(
    #     sample_weight_metrics["test_loss"]) / cfg.num_weight_samples)
    # print("Average F1-Score:", sum(
    #     sample_weight_metrics["f1_metric"]) / cfg.num_weight_samples)
    # print("Average Precision:", sum(
    #     sample_weight_metrics["precision"]) / cfg.num_weight_samples)
    # print("Average Recall:", sum(
    #     sample_weight_metrics["recall"]) / cfg.num_weight_samples)

    # print("Best Accuracy:", max(
    #     sample_weight_metrics["test_acc"]))
    # print("Best Loss:", min(
    #     sample_weight_metrics["test_loss"]))
    # print("Best F1-Score:", max(
    #     sample_weight_metrics["f1_metric"]))
    # print("Best Precision:", max(
    #     sample_weight_metrics["precision"]))
    # print("Best Recall:", max(
    #     sample_weight_metrics["recall"]))

    if cfg.log_training:
        cfg.logger.save_results(expanded_metrics,
                                "sampled_model_metrics.json")


def test_ensemble(ensemble: list, config: CONFIG):
    """
    Do a test on the ensemble of models sampled from our model generator.
    """
    ensemble_test_loss = 0.0
    ensemble_correct_preds = 0
    # ensemble = [self.model.load_state_dict(d) for d in self.model_ensemble]
    gen_model_test_dataset = DatasetRetriever(
        "MNIST")
    _, test_set = gen_model_test_dataset()
    test_dataloader = DataLoader(test_set, len(test_set))
    predictions = []
    groundtruth = []
    for _, (mbatch_x, mbatch_y) in enumerate(test_dataloader):
        mbatch_x = mbatch_x.to(config.device)
        # mbatch_y = mbatch_y.to(self.device)
        flattened_mbatch_x = torch.flatten(mbatch_x, start_dim=1)
        one_hot_mbatch_y = torch.eye(10)[
            mbatch_y].to(config.device)
        ensemble_pred_y = torch.stack(
            [m(flattened_mbatch_x) for m in ensemble], 0
        ).mean(0)
        ensemble_test_loss += F.cross_entropy(
            ensemble_pred_y, one_hot_mbatch_y)

        y_pred = torch.argmax(ensemble_pred_y.cpu(), dim=1).tolist()
        ensemble_correct_preds += torch.where(
            torch.argmax(ensemble_pred_y.cpu(), dim=1) == mbatch_y, 1, 0
        ).sum()
        predictions += y_pred
        groundtruth += mbatch_y.tolist()

    ensemble_loss = ensemble_test_loss.item()
    ensemble_acc = ensemble_correct_preds.item() / len(test_set)
    ensemble_f1_score = f1_score(
        groundtruth, predictions, average="weighted")
    ensemble_precision = precision_score(
        groundtruth, predictions, average="weighted")
    ensemble_recall = recall_score(
        groundtruth, predictions, average="weighted")
    ensemble_cm = confusion_matrix(groundtruth, predictions)
    ensemble_distinct_count = np.sum(ensemble_cm, axis=0).tolist()
    return {
        "ensemble_test_loss": ensemble_loss,
        "ensemble_test_acc": ensemble_acc,
        "ensemble_f1_metric": ensemble_f1_score,
        "ensemble_recall": ensemble_recall,
        "ensemble_precision": ensemble_precision,
        "ensemble_distinct_count": ensemble_distinct_count,
    }


if __name__ == "__main__":
    experiment_params = argument_parser()

    dataset_json_path = "/scratch_net/bmicdl03/kgolan/Thesis/datasets/small_model_dataset_MNIST/small_model_dataset.json"
    dataset_obj = PGMDataset(
        model_json_path=dataset_json_path, base_model=SmallMLP())
    weights, biases = dataset_obj()
    # print(weights[0].size(), weights[0])

    cfg = CONFIG(
        learning_rate=experiment_params.learning_rate,
        batch_size=experiment_params.batch_size,
        num_iterations=experiment_params.n_its,
        base_model=SmallMLP(),
        weight_dataset=weights,
        bias_dataset=biases,
        num_latent_samples=experiment_params.n_samples,
        num_weight_samples=experiment_params.n_samples,
        log_training=True,
    )
    run_experiment(cfg)
