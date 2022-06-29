from datetime import datetime
from typing import List
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from model.base import BaseModel
from torch.optim.lr_scheduler import ExponentialLR


class LstmModel(BaseModel):
    def __init__(self, tag: str, channels: int = 2, lr: float = 1e-3, lr_decay: float = 9e-1, 
                 adam_betas: List[float] = [9e-1, 999e-3], input_sequence: int = 1, output_sequence: int = 1,
                 log: bool = True) -> None:
        # initialize components using the parent class
        super(LstmModel, self).__init__(tag, log)

        # define hyperparameters for the network itself
        self.__channels = channels
        self.__in_seq_len = input_sequence
        self.__out_seq_len = output_sequence

        self.__net_midi = nn.Sequential(
            nn.Linear(65, 48),
            nn.LeakyReLU(),
            nn.Linear(48, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
        )

        self.__net_wave = nn.Sequential(
            nn.Linear(2, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 16)
        )

        self.__lstm = nn.LSTM(32, 128, num_layers=2, dropout=0.1, bidirectional=False, batch_first=True)
        self.__reduction = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 96),
            nn.LeakyReLU(),
            nn.Linear(96, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, self.__channels * self.__out_seq_len)
        )

        # define loss function, optimizer and scheduler for the learning rate
        self._loss_fn = torch.nn.L1Loss()
        self._optim = torch.optim.AdamW(self.parameters(), lr=lr, betas=adam_betas)
        self._scheduler = ExponentialLR(self._optim, gamma=lr_decay)
        self._cache = torch.zeros((self.__in_seq_len, 2), dtype=torch.float32)
    
    def load(self, path) -> None:
        """Loads the model's parameter given a path
        """
        self.load_state_dict(torch.load(path))
        self.eval()

    def forward(self, x) -> torch.tensor:
        x_midi, x_wave = x
        batch_size, seq_len, midi_f = x_midi.shape
        batch_size, seq_len, wave_f = x_wave.shape

        # extract features of midi/wave input
        x_midi = x_midi.view(-1, midi_f)
        x_wave = x_wave.view(-1, wave_f)
        x_midi = self.__net_midi(x_midi)
        x_wave = self.__net_wave(x_wave)
        x_midi = x_midi.view(batch_size, seq_len, 16)
        x_wave = x_wave.view(batch_size, seq_len, 16)

        # pass both extracted feature types into the LSTM
        x = torch.concat((x_midi, x_wave), dim=-1)
        x, (h, c) = self.__lstm(x)

        x = h[0]
        x = self.__reduction(x)
        x = x.view(batch_size, self.__out_seq_len, self.__channels)

        return x