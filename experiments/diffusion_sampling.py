import os, torch
from utils.params import argument_parser
from utils.profile import profile
from models.ddpm import DDPM
from models.unet import DDPMUNet
from models.diffusion_sampler import DDPMSampler

class CONFIG:
    def __init__(self, unet_checkpoint_path: str, sample_channels: int, sample_size: tuple, experiment_dir:str) -> None:
        self.unet_checkpoint_path = unet_checkpoint_path
        # self.ddpm = ddpm
        self.sample_channels = sample_channels
        self.sample_size = sample_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.experiment_dir = experiment_dir



def run(cfg: CONFIG):
    unet_checkpoint = torch.load(cfg.unet_checkpoint_path)
    unet = DDPMUNet(is_attention=[False, False, False, True])
    unet.load_state_dict(unet_checkpoint["model_state_dict"])
    ddpm = DDPM(unet, 1000, cfg.device)

    diffusion_sampler = DDPMSampler(ddpm, cfg.sample_channels, cfg.sample_size, cfg.device, cfg.experiment_dir)
    diffusion_sampler.sample(30)

if __name__ == "__main__":
    experiment_params = argument_parser()
    cfg = CONFIG(unet_checkpoint_path=experiment_params.checkpoint_file_path, sample_channels = 1, sample_size=(32,32), experiment_dir=experiment_params.experiment_dir_path)
    run(cfg) 