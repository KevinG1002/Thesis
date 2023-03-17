import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets.model_dataset import ModelsDataset
from datasets.get_dataset import DatasetRetriever
from models.ddpm import DDPM
from utils.exp_logging import checkpoint, Logger
from models.unet import DDPMUNet
from utils.profile import profile
from utils.weight_transformations import nn_to_2d_tensor, tensor_to_nn
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix
from frameworks.sgd_template import SupervisedLearning


class DDPMDiffusion:
    """
    DDPM Diffusion class.
    - ddpm: The actual diffusion model that will learn the underlying probability distribution of the dataset you want to synthesize samples from.
    - noise_predictor: model that learns to predict the noise added to a sample durring forward diffusion.
    """

    ddpm: DDPM
    noise_predictor: DDPMUNet
    optimizer: Adam

    def __init__(
        self,
        diffusion_steps: int,
        sample_channels: int,
        num_channels: int,
        sample_dimensions: "tuple[int]",
        channel_multipliers: "list[int]",
        is_attention: "list[bool]",
        n_blocks: int,
        num_gen_samples: int,
        batch_size: int,
        grad_accumulation: int,
        learning_rate: float,
        epochs: int,
        dataset: Dataset,
        device: str,
        checkpoint_every: int,
        checkpoint_dir_path: str,
        logger: Logger = None,
    ) -> None:
        """
        Diffusion Model template for training and generation of samples belonging to some dataset.

        This template takes two models as input: the DDPM (Denoising Diffusion Probabilistic Model)
        and a Noise Predictor, sometimes known as "epsilon model", that predicts the noise that has
        been added to samples distorted by the diffusion process.

        A nice explanation for how these two need to interact with each other can be found here:
        https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

        The following is a summary of the necessary arguments for this process to run.
        Args:
            - sample_channels: number of channels in a given sample in your dataset.
            - num_channels: number of channels for first projection of sample in UNet
            - sample_dimensions: shape of a given sample passed to the model.
            - channel_multipliers: list denoting how to multiply channels at different resolutions in UNet
            - is_attention: list of booleans denoting if an attention block should be used at a resolution level in a UNet.
            - num_gen_samples: number of samples to generate once diffusion process is done.
            - batch_size: batch_size during training of UNet model
            - grad_accumulation: how many batch gradients to accumulate (enables gradient computation over larger number of sampels)
            - learning_rate: learning rate for UNEt optimizer
            - epochs: number of epochs for UNet to iterate over dataset.
            - dataset: dataset to synthesize examples from.


        """
        self.diffusion_steps = diffusion_steps
        self.sample_channels = sample_channels
        self.num_channels = num_channels
        self.sample_dimensions = sample_dimensions
        self.channel_multipliers = channel_multipliers
        self.is_attention = is_attention
        self.num_gen_samples = num_gen_samples
        self.batch_size = batch_size
        self.grad_accumulation = grad_accumulation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset = dataset
        self.device = device
        self.noise_predictor = DDPMUNet(
            sample_channels, num_channels, channel_multipliers, is_attention, n_blocks
        ).to(self.device)
        self.ddpm = DDPM(self.noise_predictor,
                         self.diffusion_steps, self.device)
        self.optimizer = Adam(
            self.noise_predictor.parameters(), self.learning_rate)
        self.dataloader = DataLoader(
            self.dataset, self.batch_size, shuffle=True
        )
        # Checkpointing attributes
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir_path = checkpoint_dir_path
        self.logger = logger
        assert (checkpoint_dir_path and checkpoint_every) or not (
            checkpoint_dir_path and checkpoint_every
        ), "Missing either one of checkpoint dir (str path) or checkpoint frequency (int)"
        if checkpoint_every:
            assert (
                checkpoint_every <= self.epochs
            ), "Checkpoint frequency greater than number of epochs. Current program won't checkpoint models."
        # Attribute set up within init function.
        if self.checkpoint_dir_path:
            self.experiment_dir = os.path.dirname(self.checkpoint_dir_path)
        else:
            self.experiment_dir = os.getcwd()

    def train_epoch(self, epoch_idx):
        freq = 50 if isinstance(self.dataset, ModelsDataset) else 100
        for idx, (mbatch_x, mbatch_y) in enumerate(self.dataloader):
            mbatch_x = mbatch_x.to(self.device)
            mbatch_y = mbatch_y.to(self.device)
            self.loss = self.ddpm.l_simple(mbatch_x) / self.grad_accumulation
            if idx % freq == 0:
                print("Epoch %d : Diffusion Loss %.3f" %
                      (epoch_idx, self.loss))
            self.loss.backward()
            if (idx + 1) % self.grad_accumulation:
                self.optimizer.step()
                self.optimizer.zero_grad()

    def train(self):
        """
        Train UNet in conjuction with DDPM.
        """
        self.train_metrics = {"train_diff_loss": [],
                              "mean_test_loss": [],
                              "mean_test_acc": [],
                              "mean_f1_metric": [],
                              "mean_recall": [],
                              "mean_precision": [],
                              "mean_distinct_count": [],
                              "ensemble_test_loss": [],
                              "ensemble_test_acc": [],
                              "ensemble_f1_metric": [],
                              "ensemble_recall": [],
                              "ensemble_precision": [],
                              "ensemble_distinct_count": [],
                              }

        for epoch in range(self.epochs):
            print("\nEpoch %d\n" % (epoch + 1))
            # self.train_epoch(epoch + 1)
            self.train_epoch(epoch + 1)
            _, sample_eval_metrics, sample_eval_avg_metrics = self.sample("")
            for key in sample_eval_avg_metrics.keys():
                self.train_metrics[key].append(sample_eval_avg_metrics[key])

            self.train_metrics["train_diff_loss"].append(self.loss.item())
            if self.logger:
                self.logger.save_results(
                    sample_eval_metrics, f"sample_model_metrics_epoch{epoch+1}.json")
                self.logger.save_results(
                    self.train_metrics, f"train_metrics_so_far.json")
            if self.checkpoint_every and (epoch % self.checkpoint_every == 0):
                checkpoint_name = "ddpm_checkpoint_e_%d_loss_%.3f.pt" % (
                    (epoch + 1),
                    self.loss,
                )
                checkpoint_path = os.path.join(
                    self.checkpoint_dir_path, checkpoint_name
                )
                checkpoint(
                    checkpoint_path,
                    epoch + 1,
                    self.noise_predictor.state_dict(),
                    self.optimizer.state_dict(),
                    self.loss,
                )
        if self.checkpoint_every:
            checkpoint_name = "ddpm_fully_trained_e_%d_loss_%.3f.pt" % (
                self.epochs,
                self.loss,
            )
            checkpoint_path = os.path.join(
                self.checkpoint_dir_path, checkpoint_name)
            checkpoint(
                checkpoint_path,
                self.epochs,
                self.noise_predictor.state_dict(),
                self.optimizer.state_dict(),
                self.loss,
            )
        return self.train_metrics

    @torch.no_grad()
    def sample(self, title: str = ""):
        """
        Sample from diffusion model
        """
        if isinstance(self.dataset, ModelsDataset):
            self.sample_dimensions = self.dataset.tensor_sample_dim[1:]
        x = torch.randn(
            [
                self.num_gen_samples,
                self.sample_channels,
                self.sample_dimensions[0],
                self.sample_dimensions[1],
            ]
        )  # Sample from Standard Gaussian (distribution at end of diffusion process) in the dimensions of original sample and sample according to the number of samples to generate.

        for t_ in range(self.diffusion_steps):
            t = self.diffusion_steps - t_ - 1
            x = self.ddpm.sample_p_t_reverse_process(
                x, x.new_full((self.num_gen_samples,), t, dtype=torch.long)
            )
        # sample1, sample2, sample3, sample4, sample5 = torch.chunk(x_t, 5, 0)
        # x = x.cpu().numpy()
        # print(x.size())
        # restored_samples = []
        # for i in range(self.num_gen_samples):
        #     if isinstance(self.dataset, ModelsDataset):
        #         samples = torch.chunk(x, self.num_gen_samples, 0)
        #         restored_samples.append(
        #             self.dataset.restore_original_tensor(samples[i]))
        #     else:
        #         plt.imsave(
        #             f"{self.experiment_dir}/{title}_gen_sample_{i}.png",
        #             np.squeeze(x[i].cpu().numpy()),
        # )
        # return restored_samples
        if isinstance(self.dataset, ModelsDataset):
            sample_eval_results, sample_avg_results = self.eval_gen_model(x)
            samples = torch.chunk(x, self.num_gen_samples, 0)
            return [
                self.dataset.restore_original_tensor(samples[i])
                for i in range(self.num_gen_samples)
            ], sample_eval_results, sample_avg_results
        else:
            for i in range(self.num_gen_samples):
                plt.imsave(
                    f"{self.experiment_dir}/{title}_gen_sample_{i}.png",
                    np.squeeze(x[i].cpu().numpy()),
                )
            return

    def eval_gen_model(self, gen_models: torch.Tensor):
        samples = torch.chunk(gen_models, self.num_gen_samples, 0)
        assert isinstance(
            self.dataset, ModelsDataset
        ), "Can't evaluate generated model because it's not a tensor that can be structured as an MLP."
        gen_model_test_dataset = DatasetRetriever(
            self.dataset.original_dataset)
        _, test_set = gen_model_test_dataset()
        eval_results = {
            "test_loss": [],
            "test_acc": [],
            "f1_metric": [],
            "recall": [],
            "precision": [],
            "distinct_count": [],
        }
        # eval_mean_results = {
        #     "mean_loss" : None,
        #     "mean_acc" : None,
        #     "mean_f1" : None,
        #     "mean_recall" : None,
        #     "mean_precision" : None,
        #     "mean_count" : None
        # }
        ensemble = []
        for i in range(self.num_gen_samples):
            nn_tensor = self.dataset.restore_original_tensor(samples[i])
            nn = tensor_to_nn(nn_tensor, self.dataset.base_model)
            print(
                "\nEvaluating DDPM Generated Model %d on %s"
                % (i + 1, self.dataset.original_dataset)
            )
            mnist_process = SupervisedLearning(
                nn, test_set=test_set, device=self.device
            )
            ensemble.append(copy.deepcopy(nn))

            sample_result = mnist_process.test()
            for key in eval_results.keys():
                eval_results[key].append(sample_result[key])
        eval_avg_result = {}
        for k, v in eval_results.items():
            if type(v[0]) != list:
                eval_avg_result[f"mean_{k}"] = sum(v)/len(v)
            else:
                eval_avg_result[f"mean_{k}"] = np.mean(
                    np.array(v), axis=0).tolist()
        ensemble_metrics = self.test_ensemble(ensemble)
        eval_avg_result.update(ensemble_metrics)
        return eval_results, eval_avg_result

    def test_ensemble(self, ensemble: list):
        """
        Do a test on the ensemble of models sampled from our model generator.
        """
        ensemble_test_loss = 0.0
        ensemble_correct_preds = 0
        # ensemble = [self.model.load_state_dict(d) for d in self.model_ensemble]
        gen_model_test_dataset = DatasetRetriever(
            self.dataset.original_dataset)
        _, test_set = gen_model_test_dataset()
        test_dataloader = DataLoader(test_set, len(test_set))
        predictions = []
        groundtruth = []
        for _, (mbatch_x, mbatch_y) in enumerate(test_dataloader):
            mbatch_x = mbatch_x.to(self.device)
            # mbatch_y = mbatch_y.to(self.device)
            flattened_mbatch_x = torch.flatten(mbatch_x, start_dim=1)
            one_hot_mbatch_y = torch.eye(10)[
                mbatch_y].to(self.device)
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
        ensemble_acc = ensemble_correct_preds / len(test_set)
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
            "ensemble_test_acc": ensemble_acc.item(),
            "ensemble_f1_metric": ensemble_f1_score,
            "ensemble_recall": ensemble_recall,
            "ensemble_precision": ensemble_precision,
            "ensemble_distinct_count": ensemble_distinct_count,
        }
