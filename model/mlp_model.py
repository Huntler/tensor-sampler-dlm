from datetime import datetime
from typing import List
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from model.base import BaseModel
from torch.optim.lr_scheduler import ExponentialLR


class MlpModel(BaseModel):
    def __init__(self, input_size: int, channels: int = 2, out_act: str = "relu",
                 lr: float = 1e-3, lr_decay: float = 9e-1, adam_betas: List[float] = [9e-1, 999e-3],
                 sequence_length: int = 1, log: bool = True, precision: torch.dtype = torch.float32) -> None:
        # if logging enalbed, then create a tensorboard writer, otherwise prevent the
        # parent class to create a standard writer
        if log:
            now = datetime.now()
            self.__tb_sub = now.strftime("%d%m%Y_%H%M%S")
            self._tb_path = f"runs/MlpModel/{self.__tb_sub}"
            self._writer = SummaryWriter(self._tb_path)
        else:
            self._writer = False

        # initialize components using the parent class
        super(MlpModel, self).__init__()

        self.__input_size = input_size
        self.__channels = channels
        self.__sequence_length = sequence_length
        self.__output_activation = out_act
        self.__precision = precision
        layer_args = {"dtype": self.__precision, "device": self._device}

        midi_size = self.__input_size * self.__sequence_length
        self.__midi_input = torch.nn.Linear(midi_size, 1024, **layer_args)

        wave_size = self.__channels * self.__sequence_length
        self.__wave_input = torch.nn.Linear(wave_size, 512, **layer_args)

        self.__hidden_1 = torch.nn.Linear(1536, 768, **layer_args)
        self.__hidden_2 = torch.nn.Linear(768, 256, **layer_args)
        self.__hidden_3 = torch.nn.Linear(256, 64, **layer_args)
        self.__hidden_4 = torch.nn.Linear(64, self.__channels, **layer_args)

        # define loss function, optimizer and scheduler for the learning rate
        self._loss_fn = torch.nn.MSELoss()
        self._optim = torch.optim.AdamW(self.parameters(), lr=lr, betas=adam_betas)
        self._scheduler = ExponentialLR(self._optim, gamma=lr_decay)

        # for prediction
        self._cache = torch.zeros((self.__sequence_length, self.__channels))

    def reset_cache(self) -> None:
        self._cache_index = 0
        self._cache = torch.zeros((self.__sequence_length, self.__channels))
    
    def load(self, path) -> None:
        """Loads the model's parameter given a path
        """
        self.load_state_dict(torch.load(path))
        self.eval()

    def forward(self, X) -> torch.tensor:
        midi, wave = X
        midi = torch.flatten(midi, 1, 2)
        wave = torch.flatten(wave, 1, 2)

        # send midi and wave inputs to their first layer an concat both outputs
        x1 = self.__midi_input(midi)
        x2 = self.__wave_input(wave)
        x = torch.cat((x1, x2), dim=1)
        x = torch.relu(x)

        # pass the concat through the other dense layers
        x = self.__hidden_1(x)
        x = torch.relu(x)

        x = self.__hidden_2(x)
        x = torch.relu(x)

        x = self.__hidden_3(x)
        x = torch.relu(x)

        x = self.__hidden_4(x)

        return x