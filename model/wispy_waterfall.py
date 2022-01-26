import torch
from data.iterators import MAX_N_NOTES
from model.base import BaseModel
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class WispyWaterfall(BaseModel):
    """
    This implementation of our Model uses classic fully connected dense layers
    in order to predict the wave form given the midi files. Also, the rolling
    window size is fixed to a size of 256 samples.
    This will be pretty bad.

    Args:
        BaseModel (nn.Module): The base model.
    """

    def __init__(self) -> None:
        super().__init__(rolling_window_size=256)
        now = datetime.now()
        self._writer = SummaryWriter("runs/WispyWaterfall/" + now.strftime("%m_%d_%Y"))

        # the model's layers, optimizers, schedulers and more
        # are defined here
        self._l1 = torch.nn.Linear(256 * MAX_N_NOTES, 2048)
        self._l2 = torch.nn.Linear(2048, 1024)
        self._l3 = torch.nn.Linear(1024, 256)
        self._l4 = torch.nn.Linear(256, 32)
        self._l5 = torch.nn.Linear(32, 2)

        self._loss_fn = torch.nn.MSELoss()
        self._optim = torch.optim.AdamW(self.parameters())

    def forward(self, x):
        x = torch.flatten(x)

        x = self._l1(x)
        x = torch.relu(x)
        x = self._l2(x)
        x = torch.relu(x)
        x = self._l3(x)
        x = torch.relu(x)
        x = self._l4(x)
        x = torch.relu(x)
        x = self._l5(x)
        x = torch.sigmoid(x)

        return x
