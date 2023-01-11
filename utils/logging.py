import json
import os
import torch


class Logger:
    def __init__(
        self, experiment_name: str, experiment_dir: str, experiment_config: dict
    ):
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        self.experiment_config = experiment_config
        self._save_config()
        self._create_checkpoint_dir()

    def _save_config(self):
        with open(self.experiment_dir + "/experiment_config.json", "w") as f:
            json.dump(self.experiment_config, f)
            return

    def _create_checkpoint_dir(self):
        self.checkpoint_path = self.experiment_dir + "/checkpoints"
        if os.path.exists(self.checkpoint_path):
            return
        os.mkdir(self.checkpoint_path)
        return

    def save_results(self, dict_results: dict):
        if os.path.exists(self.experiment_dir + "/experiment_results.json"):
            with open(self.experiment_dir + "/experiment_results.json") as curr_json:
                curr_results = json.loads(curr_json.read())
                curr_metrics = set(curr_results.keys())
                new_metrics = set(dict_results.keys()).difference(curr_metrics)
                for metric in list(new_metrics):
                    curr_results[metric] = dict_results[metric]
                with open(self.experiment_dir + "/experiment_results.json", "w") as f:
                    json.dump(curr_results, f)
                    return
        with open(self.experiment_dir + "/experiment_results.json", "w") as f:
            json.dump(dict_results, f)
            return


def checkpoint(
    path: str,
    epoch: int,
    model_state_dict: dict,
    optimizer_state_dict: dict,
    loss: torch.Tensor,
):
    torch.save(
        {
            "epochs": epoch,
            "unet_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "loss": loss,
        },
        path,
    )
    return
