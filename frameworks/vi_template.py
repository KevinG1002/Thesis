from .basic_template import BasicLearning
from .sgd_template import SupervisedLearning
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.classification import MulticlassCalibrationError


class VITemplate(SupervisedLearning):
    def __init__(
        self,
        model: nn.Module,
        train_set: Dataset,
        test_set: Dataset,
        val_set: Dataset,
        num_classes: int,
        batch_size: int,
        num_mc_samples: int,
        epochs: int,
        optim: Optimizer,
        likelihood_criterion: _Loss,
        device: str,
    ):
        self.num_mc_samples = num_mc_samples
        self.likelihood_criterion = likelihood_criterion
        super(VITemplate, self).__init__(
            model,
            train_set,
            test_set,
            val_set,
            num_classes,
            epochs,
            batch_size,
            optim,
            likelihood_criterion,
        )
        self.device = device

    def train(self):
        self.val_dataloader = self._instantiate_val_dataloader()
        self.train_dataloader = self._instantiate_train_dataloader()

        self.model.train()  # Set model to train mode
        for epoch in range(self.epochs):
            loss = 0.0
            print("\nTraining for epoch %d" % (epoch + 1))
            for idx, (mbatch_x, mbatch_y) in enumerate(self.train_dataloader):
                mbatch_x = mbatch_x.to(self.device)
                mbatch_y = mbatch_y.to(self.device)
                one_hot_mbatch_y = torch.eye(self.num_classes)[mbatch_y]
                self.optim.zero_grad()
                logits, log_prior, log_var_posterior = self.model(
                    torch.flatten(mbatch_x, 1)
                )

                nll = self.likelihood_criterion(logits, one_hot_mbatch_y)
                log_odds_ratio = log_var_posterior - log_prior
                loss = nll + (1 / len(self.train_dataloader)) * log_odds_ratio
                loss.backward()
                self.optim.step()
                correct_preds_cnt = torch.where(
                    torch.argmax(logits, 1) == mbatch_y, 1, 0
                ).sum()

                if idx % 100 == 0:
                    print(
                        "Epoch %d metrics -- Batch weighted loss: %.3f || NLL: %.3f || Log-odds: %.3f || Batch accuracy: %.3f"
                        % (
                            epoch + 1,
                            loss,
                            nll,
                            log_odds_ratio,
                            correct_preds_cnt / mbatch_y.size()[0],
                        )
                    )

            with torch.no_grad():
                val_loss = 0.0
                correct_preds = 0
                for mbatch_x, mbatch_y in self.val_dataloader:
                    mbatch_x = mbatch_x.to(self.device)
                    mbatch_y = mbatch_y.to(self.device)
                    one_hot_mbatch_y = torch.eye(self.num_classes)[mbatch_y]
                    logits, log_prior, log_var_posterior = self.model(
                        torch.flatten(mbatch_x, 1)
                    )
                    nll = self.likelihood_criterion(logits, one_hot_mbatch_y)
                    log_odds_ratio = log_var_posterior - log_prior
                    val_loss += nll + (1 / len(self.train_dataloader)) * log_odds_ratio
                    correct_preds += torch.where(
                        torch.argmax(logits, 1) == mbatch_y, 1, 0
                    ).sum()
                print(
                    "Validation metrics after epoch %d -- Accuracy: %.3f || Loss: %.3f"
                    % (
                        epoch + 1,
                        correct_preds / len(self.test_set),
                        val_loss / len(self.val_dataloader),
                    )
                )

    def evaluate(self):
        self.test_dataloader = self._instantiate_test_dataloader()
        predicted_batch_probabilities = []
        with torch.no_grad():
            for mbatch_x, mbatch_y in self.test_dataloader:
                mbatch_x = mbatch_x.to(self.device)
                predict_probabilities = self.model.predict(
                    torch.flatten(mbatch_x, 1), 15
                )
                predicted_batch_probabilities.append(predict_probabilities)

            predicted_batch_probabilities = torch.cat(predicted_batch_probabilities, 0)
            predicted_classes = torch.argmax(predicted_batch_probabilities, dim=1)
            actual_classes = self.test_dataloader.dataset.test_labels
            accuracy = torch.where(
                predicted_classes == actual_classes, 1, 0
            ).sum() / len(self.test_set)
            ece_score = MulticlassCalibrationError(
                num_classes=self.num_classes, n_bins=30, norm="l1"
            )
        print(
            "Test set metrics -- Accuracy: %.3f || ECE score: %.3f"
            % (accuracy, ece_score(predicted_batch_probabilities, actual_classes))
        )
