import torch
from data.iterators import MAX_N_NOTES
from model.base import BaseModel
from datetime import datetime
import numpy as np
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

    def __init__(self, log: bool=True) -> None:
        # if logging enalbed, then create a tensorboard writer, otherwise prevent the
        # parent class to create a standard writer
        if log:
            now = datetime.now()
            self.__tb_sub = now.strftime("%d%m%Y_%H%M%S")
            self.__tb_path = f"runs/WispyWaterfall/{self.__tb_sub}"
            self._writer = SummaryWriter(self.__tb_path)
        else:
            self._writer = False
        
        # initialize components using the parent class
        super().__init__()

        # the model's layers, optimizers, schedulers and more
        # are defined here
        self._l1 = torch.nn.Linear(20, 2048)
        self._l2 = torch.nn.Linear(2048, 1024)
        self._l3 = torch.nn.Linear(1024, 512)
        self._l4 = torch.nn.Linear(512, 256)
        self._l5 = torch.nn.Linear(256, 2)

        self._loss_fn = torch.nn.MSELoss()
        self._optim = torch.optim.AdamW(self.parameters())

    def save_to_default(self) -> None:
        model_tag = datetime.now().strftime("%H%M%S")
        params = self.state_dict()
        torch.save(params, f"{self.__tb_path}/model_{model_tag}.torch")
    
    def load(self, path) -> None:
        self.load_state_dict(torch.load(path))
        self.eval()

    def forward(self, x):
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