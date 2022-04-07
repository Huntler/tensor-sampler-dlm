from typing import Tuple
import torch
from data.iterators import MAX_N_NOTES
from model.base import BaseModel
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class CrispyCranberry(BaseModel):
    """
    This implementation of our Model uses long short term memory (LSTM)
    in order to predict the wave form given the midi files. Also, the rolling
    window size is fixed to a size of 256 samples.
    This will be pretty bad.

    Args:
        BaseModel (nn.Module): The base model.
    """

    def __init__(self, log: bool = True, precision: torch.dtype = torch.float16) -> None:
        # if logging enalbed, then create a tensorboard writer, otherwise prevent the
        # parent class to create a standard writer
        if log:
            now = datetime.now()
            self.__tb_sub = now.strftime("%d%m%Y_%H%M%S")
            self._tb_path = f"runs/CrispyCranberry/{self.__tb_sub}"
            self._writer = SummaryWriter(self._tb_path)
        else:
            self._writer = False

        # initialize components using the parent class
        super().__init__()

        # the model's layers, optimizers, schedulers and more
        # are defined here
        self.__hidden_dim = 32
        self.__precision = precision

        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = torch.nn.LSTMCell(
            20, self.__hidden_dim, dtype=self.__precision)
        self.lstm2 = torch.nn.LSTMCell(
            self.__hidden_dim, self.__hidden_dim, dtype=self.__precision)
        self.linear = torch.nn.Linear(
            self.__hidden_dim, 2, dtype=self.__precision)

        self._loss_fn = torch.nn.MSELoss()
        self._optim = torch.optim.AdamW(self.parameters())

    @property
    def precision(self) -> torch.dtype:
        return self.__precision

    def __init_hidden_states(self, n_samples: int) -> Tuple[torch.tensor]:
        h_t = torch.zeros(n_samples, self.__hidden_dim, dtype=self.__precision)
        c_t = torch.zeros(n_samples, self.__hidden_dim, dtype=self.__precision)

        return h_t, c_t

    def load(self, path) -> None:
        """Loads the model's parameter given a path
        """
        self.load_state_dict(torch.load(path))
        self.eval()

    def forward(self, y):
        outputs, n_samples = [], y.size(0)

        h_t, c_t = self.__init_hidden_states(n_samples)
        h_t2, c_t2 = self.__init_hidden_states(n_samples)

        for time_step in y.split(20, dim=1):
            # initial hidden and cell states
            h_t, _ = self.lstm1(time_step, (h_t, c_t))

            # new hidden and cell states
            h_t2, _ = self.lstm2(h_t, (h_t2, c_t2))

            # output from the last layer
            output = self.linear(h_t2)
            outputs.append(torch.tanh(output))

        # transform list to tensor
        outputs = torch.cat(outputs, dim=1)
        return outputs
