from datetime import datetime
from turtle import forward
from numpy import dtype
from torch.utils.tensorboard import SummaryWriter
import torch
from model.base import BaseModel


class LittleLion(BaseModel):
    def __init__(self, log: bool = True, precision: torch.dtype = torch.float16) -> None:
        # if logging enalbed, then create a tensorboard writer, otherwise prevent the
        # parent class to create a standard writer
        if log:
            now = datetime.now()
            self.__tb_sub = now.strftime("%d%m%Y_%H%M%S")
            self._tb_path = f"runs/LittleLion/{self.__tb_sub}"
            self._writer = SummaryWriter(self._tb_path)
        else:
            self._writer = False

        super(LittleLion, self).__init__()
        self.__n_layers = 8
        self.__hidden_dim = 256
        self.__precision = precision

        self.__gru = torch.nn.GRU(
            20, self.__hidden_dim, self.__n_layers, dropout=0.1)
        self.__linear_1 = torch.nn.Linear(self.__hidden_dim, 2)

        self._loss_fn = torch.nn.MSELoss()
        self._optim = torch.optim.AdamW(self.parameters())

    def __init_hidden_states(self, batch_size: int) -> torch.tensor:
        weight = next(self.parameters()).data
        hidden = weight.new(self.__n_layers, batch_size, self.__hidden_dim,
                            device=self._device).zero_()
        return hidden

    def load(self, path) -> None:
        """Loads the model's parameter given a path
        """
        self.load_state_dict(torch.load(path))
        self.eval()

    def forward(self, y):
        h = self.__init_hidden_states(batch_size=y.size(0))
        output = []

        for time_step in y.split(20, dim=1):
            time_step = torch.unsqueeze(time_step, 0)
            x, h = self.__gru(time_step, h)
            x = torch.relu(x[0][0])

            x = self.__linear_1(x)
            x = torch.tanh(x)
            output.append(x)

        output = torch.stack(output)
        return output
