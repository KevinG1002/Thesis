import torch, os
import torch.nn as nn
from torch.nn import MSELoss
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, random_split, DataLoader
from torch.optim import Adam
from frameworks.basic_template import BasicLearning
from utils.exp_logging import checkpoint
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from models.mlp import RegressMLP
from datasets.get_dataset import DatasetRetriever


class RegressionTemplate(BasicLearning):
    optim: Adam

    def __init__(
        self,
        model: nn.Module = None,
        train_set: Dataset = None,
        test_set: Dataset = None,
        val_set: Dataset = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        criterion: _Loss = None,
        device: str = "cpu",
        checkpoint_every: int = None,
        checkpoint_dir: str = None,
    ):
        super().__init__()
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_set = val_set if val_set else self._instantiate_val_set()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = criterion if criterion else MSELoss()
        self.device = device
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir

        assert (checkpoint_dir and checkpoint_every) or not (
            checkpoint_dir and checkpoint_every
        ), "Missing either one of checkpoint dir (str path) or checkpoint frequency (int)"
        if checkpoint_every:
            assert (
                checkpoint_every <= self.epochs
            ), "Checkpoint frequency greater than number of epochs. Current program won't checkpoint models."

    def _instantiate_val_set(self, val_split: float = 0.15):
        """
        Internal method used to create a validation set if it isn't passed as an attribute upon class instantiation.
        Validation set created by splitting training set according to val_split float.
        Validation set only created if training set is not None.
        """
        if self.train_set:
            new_train_len = int((1 - val_split) * len(self.train_set))
            val_len = int(val_split * len(self.train_set))
            self.train_set, new_val_set = random_split(
                self.train_set,
                (new_train_len, val_len),
                generator=torch.Generator().manual_seed(42),
            )
            return new_val_set
        else:
            return None

    def _instantiate_train_dataloader(self):
        return DataLoader(
            dataset=self.train_set, batch_size=self.batch_size, shuffle=True
        )

    def _instantiate_test_dataloader(self):
        return DataLoader(
            dataset=self.test_set, batch_size=self.batch_size, shuffle=True
        )

    def _instantiate_val_dataloader(self):
        return DataLoader(
            dataset=self.val_set, batch_size=self.batch_size, shuffle=True
        )

    def train(self) -> dict:
        """
        Train loop. Returns dictionary of training metrics including epoch validation loss and accuracy.
        """
        self.train_dataloader = self._instantiate_train_dataloader()
        self.val_dataloader = self._instantiate_val_dataloader()
        self.model.train(True)
        # Beginning of training loop #
        print("\n\n")
        for epoch in range(self.epochs):
            print("\nStart of training epoch %d." % (epoch + 1))
            train_ep_metrics = self.train_epoch(epoch + 1)
            print("\nTraining for epoch %d done." % (epoch + 1))
            if epoch < 1:
                train_metrics = {k: [v] for k, v in train_ep_metrics.items()}
            else:
                for k in train_ep_metrics.keys():
                    train_metrics[k].append(train_ep_metrics[k])
            if self.checkpoint_every and ((epoch + 1) % self.checkpoint_every == 0):
                checkpoint_name = "%s_checkpoint_e_%d_loss_%.3f.pt" % (
                    self.model.name,
                    epoch + 1,
                    train_ep_metrics["val_loss"],
                )
                model_checkpoint_path = os.path.join(
                    self.checkpoint_dir, checkpoint_name
                )
                checkpoint(
                    model_checkpoint_path,
                    epoch + 1,
                    self.model.state_dict(),
                    self.optimizer.state_dict(),
                    train_metrics["val_loss"],
                )
        if self.checkpoint_every:
            checkpoint_name = "%s_fully_trained_e_%d_loss_%.3f.pt" % (
                self.model.name,
                self.epochs,
                train_ep_metrics["val_loss"],
            )
            model_checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            checkpoint(
                model_checkpoint_path,
                epoch + 1,
                self.model.state_dict(),
                self.optimizer.state_dict(),
                train_metrics["val_loss"],
            )
        return train_metrics

    def train_epoch(self, epoch_idx: int):
        train_loss = 0.0
        for idx, (mbatch_x, mbatch_y) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            mbatch_x = mbatch_x.unsqueeze(-1).unsqueeze(-1).to(self.device)
            pred_y = self.model(mbatch_x).squeeze()
            # print(pred_y.size(), mbatch_y.size())
            loss = self.criterion(
                pred_y,
                mbatch_y,
            )
            if idx % 100 == 0:
                print("Training loss: %.3f" % loss)
            loss.backward()
            train_loss += loss
            self.optimizer.step()
        avg_train_loss = train_loss.item() / len(self.train_dataloader)

        # Beginning of validation loop #
        with torch.no_grad():
            val_loss = 0.0
            predictions = []
            groundtruth = []
            for idx, (mbatch_x, mbatch_y) in enumerate(self.val_dataloader):
                mbatch_x = mbatch_x.unsqueeze(-1).to(self.device)
                pred_y: torch.Tensor = self.model(mbatch_x).squeeze()
                val_loss += self.criterion(pred_y, mbatch_y)
                predictions += pred_y.tolist()
                groundtruth += mbatch_y.tolist()

            val_mse = mean_squared_error(groundtruth, predictions)
            val_mae = mean_absolute_error(groundtruth, predictions)
            val_mape = mean_absolute_percentage_error(groundtruth, predictions)
            val_r2 = r2_score(groundtruth, predictions)
            print(
                "\nEpoch %d\t Avg Train Loss: %.3f|| Avg Validation Loss: %.3f|| Validation MSE: %.3f || Validation MAE: %.3f || Validation MAPE: %.3f || Validation R2-Score: %.3f"
                % (
                    epoch_idx,
                    avg_train_loss,
                    val_loss / len(self.val_dataloader),
                    val_mse,
                    val_mae,
                    val_mape,
                    val_r2,
                )
            )
            return {
                "train_loss": avg_train_loss,
                "val_loss": val_loss.item() / len(self.val_dataloader),
                "val_mse": val_mse,
                "val_mae": val_mae,
                "val_mape": val_mape,
                "val_r2": val_r2,
            }

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.test_dataloader = self._instantiate_test_dataloader()
        test_loss = 0.0

        predictions = []
        groundtruth = []
        for mbatch_x, mbatch_y in self.test_dataloader:
            mbatch_x = mbatch_x.unsqueeze(-1).to(self.device)
            pred_y = self.model(mbatch_x).squeeze()
            test_loss += self.criterion(pred_y, mbatch_y)
            predictions += pred_y.tolist()
            groundtruth += mbatch_y.tolist()

        test_mse = mean_squared_error(groundtruth, predictions)
        test_mae = mean_absolute_error(groundtruth, predictions)
        test_mape = mean_absolute_percentage_error(groundtruth, predictions)
        test_r2 = r2_score(groundtruth, predictions)
        print(
            "\nTesting Loss: %.3f || Testing MSE: %.3f || Testing MAE: %.3f || Testing MAPE: %.3f || Testing R2-Score: %.3f"
            % (
                test_loss / len(self.test_dataloader),
                test_mse,
                test_mae,
                test_mape,
                test_r2,
            )
        )
        return {
            "test_loss": test_loss.item() / len(self.test_dataloader),
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_mape": test_mape,
            "test_r2": test_r2,
        }


def test():
    nn = RegressMLP()
    # nn.load_state_dict(
    #     torch.load(
    #         "/Users/kevingolan/Documents/Coding_Assignments/Thesis/datasets/model_dataset_MNIST/models/mlp_mnist_model_0.pth",
    #         map_location="cpu",
    #     )
    # )
    d = DatasetRetriever("SineDataset")
    train_data, test_data = d()
    process = RegressionTemplate(nn, train_data, test_data, batch_size=16)
    process.train()
    process.test()


if __name__ == "__main__":
    test()
