from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    DATASETDIRPATH = "/scratch_net/bmicdl03/kgolan/Thesis/datasets/SineDataset"
else:
    DATASETDIRPATH = (
        "/Users/kevingolan/Documents/Coding_Assignments/Thesis/datasets/SineDataset"
    )
DATASETOBJPATH = DATASETDIRPATH + "/dataset.pt"


class SineDataset(Dataset):
    def __init__(self, size: int, range: tuple[int, int]):
        self.size = size
        self.start, self.end = range
        self.step = (self.end - self.start) / self.size
        if os.path.exists(DATASETDIRPATH):
            if os.path.exists(DATASETOBJPATH):
                self.samples: torch.Tensor = torch.load(DATASETOBJPATH)
                if (
                    len(self.samples) != self.size
                    or torch.max(
                        self.samples[
                            0:,
                        ]
                    )
                    <= self.end
                ):
                    self.samples = torch.tensor(
                        [
                            (x, self.sine_gen(x))
                            for x in torch.arange(
                                self.start,
                                self.end,
                                self.step,
                            )
                        ]
                    )
                    self.save_dataset(self.samples, DATASETOBJPATH)
            else:
                self.samples = torch.tensor(
                    [
                        (x, self.sine_gen(x))
                        for x in torch.arange(self.start, self.end, self.step)
                    ]
                )
                self.save_dataset(self.samples, DATASETOBJPATH)

        else:  # First time loading dataset
            os.mkdir(DATASETDIRPATH)
            self.samples = torch.tensor(
                [
                    (x, self.sine_gen(x))
                    for x in torch.arange(self.start, self.end, self.step)
                ]
            )
            self.save_dataset(self.samples, DATASETOBJPATH)

        super(SineDataset, self).__init__()

    def sine_gen(self, x):
        eps = torch.randn((1)) / 5000
        return torch.sin(x) + eps

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return x, y

    def save_dataset(self, samples, path):
        torch.save(samples, path)


def test():
    dataset = SineDataset(151, (0, 15))
    dataloader = DataLoader(dataset, 1)
    data = []
    for x, y in dataloader:
        print(x, y)
        data.append(y.item())

    plt.plot(data)
    plt.show()


if __name__ == "__main__":
    test()
