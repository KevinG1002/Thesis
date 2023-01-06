import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, SGD


class SnapshotEnsemble:
    """
    From paper: SNAPSHOT ENSEMBLES: TRAIN 1, GET M FOR FREE (ICLR 2017)
    """

    optimizer: Adam

    def __init__(
        self,
        model: nn.Module,
        train_set: Dataset = None,
        test_set: Dataset = None,
        val_set: Dataset = None,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 1e-4,
    ):
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        # Set attributes upon initialisation
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.val_set = val_set if val_set else self._instantiate_val_set()

    def train(self):
        self.train_dataloader = self._instantiate_train_dataloader()
        self.val_dataloader = self._instantiate_val_dataloader()

        for idx, mbatch_x, mbatch_y in enumerate(self.train_dataloader):
            pass

    @torch.no_grad()
    def test(self):
        pass

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
