import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader
from torch.optim import Optimizer, SGD
from .basic_template import BasicLearning


class SupervisedLearning(BasicLearning):
    def __init__(
        self,
        model: nn.Module = None,
        train_set: Dataset = None,
        test_set: Dataset = None,
        val_set: Dataset = None,
        num_classes: int = 10,
        epochs: int = 10,
        batch_size: int = 32,
        optim: Optimizer = None,
        criterion: _Loss = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_set = val_set if val_set else self._instantiate_val_set()
        self.optim = optim if optim else SGD(self.model.parameters(), lr=0.001)
        self.criterion = criterion if criterion else CrossEntropyLoss()
        self.num_classes = num_classes
        self.device = device

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

    def train(self):
        self.train_dataloader = self._instantiate_train_dataloader()
        self.val_dataloader = self._instantiate_val_dataloader()
        self.model.train(True)
        # Beginning of training loop #
        for epoch in range(self.epochs):
            print("\n\n")
            loss = 0
            for idx, (mbatch_x, mbatch_y) in enumerate(self.train_dataloader):
                self.optim.zero_grad()
                mbatch_x = mbatch_x.to(self.device)
                mbatch_y = mbatch_y.to(self.device)
                flattened_mbatch_x = torch.flatten(mbatch_x, start_dim=1)
                one_hot_mbatch_y = torch.eye(self.num_classes)[mbatch_y]
                pred_y = self.model(flattened_mbatch_x)
                loss = self.criterion(
                    pred_y,
                    one_hot_mbatch_y,
                )
                if idx % 100 == 0:
                    print("Training loss: %.3f" % loss)
                loss.backward()
                self.optim.step()
            print(
                "\nTraining for epoch %d done. Evaluating on validation set."
                % (epoch + 1)
            )
            # Beginning of validation loop #
            with torch.no_grad():
                val_loss = 0.0
                correct_preds = 0
                for idx, (mbatch_x, mbatch_y) in enumerate(self.val_dataloader):
                    mbatch_x = mbatch_x.to(self.device)
                    mbatch_y = mbatch_y.to(self.device)
                    flattened_mbatch_x = torch.flatten(mbatch_x, start_dim=1)
                    one_hot_mbatch_y = torch.eye(self.num_classes)[mbatch_y]
                    pred_y = self.model(flattened_mbatch_x)
                    val_loss += self.criterion(pred_y, one_hot_mbatch_y)
                    correct_preds += torch.where(
                        torch.argmax(pred_y, dim=1) == mbatch_y,
                        1,
                        0,
                    ).sum()
                print(
                    "\nEpoch %d\tAvg Validation Loss: %.3f|| Validation Accuracy: %.3f"
                    % (
                        epoch + 1,
                        val_loss / len(self.val_dataloader),
                        float(correct_preds / len(self.val_set)),
                    )
                )

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.test_dataloader = self._instantiate_test_dataloader()
        test_loss = 0.0
        correct_preds = 0
        for mbatch_x, mbatch_y in self.test_dataloader:
            mbatch_x = mbatch_x.to(self.device)
            mbatch_y = mbatch_y.to(self.device)
            flattened_mbatch_x = torch.flatten(mbatch_x, start_dim=1)
            one_hot_mbatch_y = torch.eye(self.num_classes)[mbatch_y]
            pred_y = self.model(flattened_mbatch_x)
            test_loss += self.criterion(pred_y, one_hot_mbatch_y)
            correct_preds += torch.where(
                torch.argmax(pred_y, dim=1) == mbatch_y, 1, 0
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
