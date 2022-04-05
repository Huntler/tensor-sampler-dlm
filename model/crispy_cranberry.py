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

    def __init__(self, log: bool=True) -> None:
        # if logging enalbed, then create a tensorboard writer, otherwise prevent the
        # parent class to create a standard writer
        if log:
            now = datetime.now()
            self.__tb_sub = now.strftime("%d%m%Y_%H%M%S")
            self.__tb_path = f"runs/CrispyCranberry/{self.__tb_sub}"
            self._writer = SummaryWriter(self.__tb_path)
        else:
            self._writer = False
        
        # initialize components using the parent class
        super().__init__()

        # the model's layers, optimizers, schedulers and more
        # are defined here
        self.hidden_layers = 1024

        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = torch.nn.LSTMCell(20, self.hidden_layers)
        self.lstm2 = torch.nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = torch.nn.Linear(self.hidden_layers, 2)

        self._loss_fn = torch.nn.MSELoss()
        self._optim = torch.optim.AdamW(self.parameters())

    def save_to_default(self) -> None:
        model_tag = datetime.now().strftime("%H%M%S")
        params = self.state_dict()
        torch.save(params, f"{self.__tb_path}/model_{model_tag}.torch")
    
    def load(self, path) -> None:
        self.load_state_dict(torch.load(path))
        self.eval()

    def forward(self, y, future_preds=0):
        outputs, n_samples = [], y.size(0)

        h_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        
        for time_step in y.split(20, dim=1):
            # initial hidden and cell states
            h_t, c_t = self.lstm1(time_step, (h_t, c_t))

            # new hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))

            # output from the last layer
            output = self.linear(h_t2)
            outputs.append(torch.tanh(output))

        # transform list to tensor    
        outputs = torch.cat(outputs, dim=1)
        return outputs