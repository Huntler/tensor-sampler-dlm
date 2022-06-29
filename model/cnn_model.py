from datetime import datetime
from typing import List
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from model.base import BaseModel
from torch.optim.lr_scheduler import ExponentialLR


class CnnModel(BaseModel):
    def __init__(self, tag: str, channels: int = 2, lr: float = 1e-3, lr_decay: float = 9e-1, 
                 adam_betas: List[float] = [9e-1, 999e-3], input_sequence: int = 1, 
                 output_sequence: int = 1, log: bool = True) -> None:

        # initialize components using the parent class
        super(CnnModel, self).__init__(tag, log)

        # define hyperparameters for the network itself
        self.__channels = channels
        self._cache = np.array((input_sequence, channels))
        self.__sequence_length = input_sequence
        self.__out_sequence = output_sequence

        # define network for midi input
        self.__net_midi = nn.Sequential(
            nn.Conv1d(self.__sequence_length, self.__sequence_length // 2, 3, 1, 1),
            nn.BatchNorm1d(self.__sequence_length // 2),
            nn.Tanh(),
            nn.Conv1d(self.__sequence_length // 2, self.__sequence_length // 4, 3, 1, 1),
            nn.BatchNorm1d(self.__sequence_length // 4),
            nn.Tanh(),
            nn.Conv1d(self.__sequence_length // 4, self.__sequence_length // 8, 3, 1, 1),
            nn.BatchNorm1d(self.__sequence_length // 8),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(self.__sequence_length // 8 * 65, 512),
            nn.LeakyReLU()
        )

        # define network for extracted midi features and wave form
        self.__net_combined = nn.Sequential(
            nn.Linear(512 + self.__sequence_length * self.__channels, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.__channels * self.__out_sequence)
        )

        # define loss function, optimizer and scheduler for the learning rate
        self._loss_fn = torch.nn.MSELoss()
        self._optim = torch.optim.AdamW(self.parameters(), lr=lr, betas=adam_betas)
        self._scheduler = ExponentialLR(self._optim, gamma=lr_decay)
    
    def load(self, path) -> None:
        """Loads the model's parameter given a path
        """
        self.load_state_dict(torch.load(path))
        self.eval()

    def forward(self, x) -> torch.tensor:
        x_midi, x_wave = x
        x_wave = torch.flatten(x_wave, 1)

        x = self.__net_midi(x_midi)
        x = torch.concat((x, x_wave), dim=-1)
        x = self.__net_combined(x)

        batch_size, _dim1 = x.shape
        x = x.view(batch_size, self.__out_sequence, self.__channels)

        return x