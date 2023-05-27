from datetime import datetime
from typing import List
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from model.base import BaseModel
from torch.optim.lr_scheduler import ExponentialLR


class CnnModel(BaseModel):
    def __init__(self, tag: str, lr: float = 1e-3, lr_decay: float = 9e-1, 
                 sequence_length: int = 1, channels: int = 2,
                 adam_betas: List[float] = [9e-1, 999e-3], log: bool = True, 
                 precision: torch.dtype = torch.float32) -> None:

        # initialize components using the parent class
        super(CnnModel, self).__init__(tag, log, precision)

        # define network holders
        cnn_output_size = (sequence_length + 2 * 1 - 1 *(10 - 1) - 1) / 1 + 1
        cnn_output_size = (cnn_output_size + 2 * 1 - 1 *(5 - 1) - 1) / 1 + 1
        self.__net_wave = nn.Sequential(
            nn.Conv1d(channels, 4, 10, 1, 1),
            nn.BatchNorm1d(4),
            nn.Tanh(),
            nn.Conv1d(4, 8, 5, 1, 1),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(int(cnn_output_size * 8), 512),
            nn.LeakyReLU()
        )

        self.__net_reduction = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, channels)
        )

        # define loss function, optimizer and scheduler for the learning rate
        self.__lr = lr
        self.__adam_betas = adam_betas
        self.__lr_decay = lr_decay

        self._loss_fn = torch.nn.MSELoss()
        self._optim = torch.optim.AdamW(self.parameters(), lr=self.__lr, betas=self.__adam_betas)
        self._scheduler = ExponentialLR(self._optim, gamma=self.__lr_decay)
    
    def load(self, path) -> None:
        """Loads the model's parameter given a path
        """
        self.load_state_dict(torch.load(path))
        self.eval()

    def forward(self, x) -> torch.tensor:
        x = torch.swapaxes(x, 1, 2)

        x = self.__net_wave(x)
        x = self.__net_reduction(x)

        return x