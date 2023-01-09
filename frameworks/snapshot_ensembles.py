import torch
import copy
import numpy as np
import torch.nn as nn
from typing import Callable
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
from torch.nn import CrossEntropyLoss


class SnapshotEnsemble:
    """
    From paper: SNAPSHOT ENSEMBLES: TRAIN 1, GET M FOR FREE (ICLR 2017)
    https://arxiv.org/abs/1704.00109
    """

    optimizer: Adam
    lr_scheduler: CosineAnnealingLR

    def __init__(
        self,
        model: nn.Module,
        train_set: Dataset = None,
        test_set: Dataset = None,
        val_set: Dataset = None,
        num_classes: int = 10,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 1e-2,
        M_snapshots: int = 10,
        criterion: Callable = CrossEntropyLoss,
    ):
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        # Set attributes upon initialisation
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=int(epochs / M_snapshots), eta_min=0, verbose=True
        )
        self.criterion = criterion
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.val_set = val_set if val_set else self._instantiate_val_set()
        self.M_snapshots = M_snapshots
        self.model_ensemble = []
        self.snapshot_model_accs = []
        self.snapshot_model_loss = []

    def train_epoch(self, epoch_id: int):
        print("\nEpoch %d:\n" % epoch_id)
        for idx, (mbatch_x, mbatch_y) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            mbatch_x = mbatch_x.to(self.device)
            flattened_mbatch_x = torch.flatten(mbatch_x, start_dim=1)
            one_hot_mbatch_y = torch.eye(self.num_classes)[mbatch_y].to(self.device)
            pred_y = self.model(flattened_mbatch_x)
            loss = self.criterion(
                pred_y,
                one_hot_mbatch_y,
            )
            if idx % 100 == 0:
                print("Training loss: %.3f" % loss)
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            val_loss = 0.0
            correct_preds = 0
            for idx, (mbatch_x, mbatch_y) in enumerate(self.val_dataloader):
                mbatch_x = mbatch_x.to(self.device)
                # mbatch_y = mbatch_y.to(self.device)
                flattened_mbatch_x = torch.flatten(mbatch_x, start_dim=1)
                one_hot_mbatch_y = torch.eye(self.num_classes)[mbatch_y].to(self.device)
                pred_y = self.model(flattened_mbatch_x)
                val_loss += self.criterion(pred_y, one_hot_mbatch_y)
                correct_preds += torch.where(
                    torch.argmax(pred_y.cpu(), dim=1) == mbatch_y,
                    1,
                    0,
                ).sum()
            print(
                "\nEpoch %d: Avg Validation Loss: %.3f|| Validation Accuracy: %.3f"
                % (
                    epoch_id,
                    val_loss / len(self.val_dataloader),
                    float(correct_preds / len(self.val_set)),
                )
            )

    def train(self):
        self.train_dataloader = self._instantiate_train_dataloader()
        self.val_dataloader = self._instantiate_val_dataloader()
        snapshot_counter = 0
        for epoch in range(self.epochs):
            self.train_epoch(epoch + 1)
            self.lr_scheduler.step()
            if (epoch + 1) % (self.epochs / self.M_snapshots) == 0:
                self.model_ensemble.append(copy.deepcopy(self.model))
                snapshot_counter += 1
                (
                    loss,
                    acc,
                ) = (
                    self.test_model().values()
                )  # Test model on testing set after resetting cyclical learning rate
                self.snapshot_model_accs.append(acc)
                self.snapshot_model_loss.append(loss.item())

    @torch.no_grad()
    def test_model(self):
        self.test_dataloader = self._instantiate_test_dataloader()
        test_loss = 0.0
        correct_preds = 0
        for _, (mbatch_x, mbatch_y) in enumerate(self.test_dataloader):
            mbatch_x = mbatch_x.to(self.device)
            flattened_mbatch_x = torch.flatten(mbatch_x, start_dim=1)
            one_hot_mbatch_y = torch.eye(self.num_classes)[mbatch_y].to(self.device)
            pred_y = self.model(flattened_mbatch_x)
            test_loss += self.criterion(pred_y, one_hot_mbatch_y)
            correct_preds += torch.where(
                torch.argmax(pred_y.cpu(), dim=1) == mbatch_y, 1, 0
            ).sum()

        print(
            "\nTesting Loss: %.3f || Testing Accuracy: %.3f"
            % (
                test_loss / len(self.test_dataloader),
                float(correct_preds / len(self.test_set)),
            )
        )
        return {
            "test_loss": test_loss / len(self.test_dataloader),
            "accuracy": float(correct_preds / len(self.test_set)),
        }

    def test_ensemble(self):
        ensemble_test_loss = 0.0
        ensemble_correct_preds = 0
        # ensemble = [self.model.load_state_dict(d) for d in self.model_ensemble]
        for _, (mbatch_x, mbatch_y) in enumerate(self.test_dataloader):
            mbatch_x = mbatch_x.to(self.device)
            # mbatch_y = mbatch_y.to(self.device)
            flattened_mbatch_x = torch.flatten(mbatch_x, start_dim=1)
            one_hot_mbatch_y = torch.eye(self.num_classes)[mbatch_y].to(self.device)
            ensemble_pred_y = torch.stack(
                [m(flattened_mbatch_x) for m in self.model_ensemble], 0
            ).mean(0)
            ensemble_test_loss += self.criterion(ensemble_pred_y, one_hot_mbatch_y)
            ensemble_correct_preds += torch.where(
                torch.argmax(ensemble_pred_y.cpu(), dim=1) == mbatch_y, 1, 0
            ).sum()

        print(
            "\nEnsemble Testing Loss: %.3f || Ensemble Testing Accuracy: %.3f"
            % (
                ensemble_test_loss / len(self.test_dataloader),
                float(ensemble_correct_preds / len(self.test_set)),
            )
        )
        print(
            "Snapshot Models Average Loss: %.3f || Snapshot Models Average Accuracy: %.3f"
            % (
                float(sum(self.snapshot_model_loss) / len(self.snapshot_model_accs)),
                float(sum(self.snapshot_model_accs) / len(self.snapshot_model_accs)),
            )
        )
        return {
            "test_loss": ensemble_test_loss / len(self.test_dataloader),
            "accuracy": float(ensemble_correct_preds / len(self.test_set)),
        }

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
