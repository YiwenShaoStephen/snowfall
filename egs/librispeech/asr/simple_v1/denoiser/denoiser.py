# Author : Sonal Joshi, based on intial scripts by Saurabh Kataria

import logging
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import torch
import yaml
from .conv_tasnet import TasNet
from sklearn.preprocessing import StandardScaler
from torch import nn

torch.autograd.set_detect_anomaly(True)

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class DenoiserReconstruction(nn.Module):
    def __init__(self,  denoiser, device):
        super().__init__()
        self.denoiser = denoiser
        self.device = device

    def forward(self, audio):
        """
        Passing wavefrom through Denoiser Defender.
        """

        # Added for processing single audio file as in deepspeech armory [Sonal 29Oct20]
        if audio.ndim == 1:
            num_samples = audio.shape[0]
            #recording = audio.detach().cpu().numpy()
            recording = audio.unsqueeze(dim=0).to(self.device)
            #recording = audio.unsqueeze(dim=0)

            # Setup inputs
            reconstructed_audio = self.denoiser(recording)
            #return reconstructed_audio.squeeze()
            return reconstructed_audio.squeeze().to(self.device)

        else:
            return self.denoiser(audio).squeeze(1) # B, T
            # reconstructions = []
            # num_samples = audio.shape[1]
            # for idx in range(audio.shape[0]):
            #     #recording = audio[idx, :].detach().cpu().numpy()
            #     #recording = audio.detach().cpu().unsqueeze(dim=0)
            #     #recording = audio.unsqueeze(dim=0)
            #     recording = audio.unsqueeze(dim=0).to(self.device)

            #     # Setup inputs
            #     reconstructed_audio = self.denoiser(recording)
            #     #reconstructed_audio = reconstructed_audio.squeeze()
            #     reconstructed_audio = reconstructed_audio.squeeze().to(self.device)
            #     reconstructions.append(reconstructed_audio)

            # #return torch.stack(reconstructions) 
            # return torch.stack(reconstructions, device=self.device)

class DenoiserDefender(nn.Module):
    def __init__(self, denoiser_model_dir: Path, denoiser_model_ckpt: Path,  device: str):
        super().__init__()
        self.device = device
        # with open(denoiser_model_dir / 'config.yml') as f:
        #     self.config = yaml.load(f, Loader=yaml.Loader)

        self.model = TasNet(num_spk=1, layer=8, enc_dim=128, stack=1, kernel=3, win=1, TCN_dilationFactor=2)
        # self.model = TasNet(num_spk=self.config['num_spk'], layer=self.config['layer'], enc_dim=self.config['enc_dim'], stack=self.config['stack'], kernel=self.config['kernel'], win=self.config['win'], TCN_dilationFactor=self.config['TCN_dilationFactor'])

        # model_path = denoiser_model_dir / denoiser_model_ckpt
        model_path = denoiser_model_ckpt

        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
        # load_string = self.config["load_model_string"]
        # self.model.load_state_dict(torch.load(model_path, map_location=self.device)[load_string])
        self.model.to(self.device)

        self.reconstructor = DenoiserReconstruction(self.model, self.device)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.reconstructor(audio)
