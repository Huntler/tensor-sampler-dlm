import contextlib
from datetime import datetime
from typing import List, Tuple
import numpy as np
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm


@contextlib.contextmanager
def no_autocast():
    yield None


class BaseModel(nn.Module):
    def __init__(self, tag: str, log: bool = True) -> None:
        """
        Base class of a model. This only includes method-playeholders 
        and general methods for statistics etc.
        """
        super(BaseModel, self).__init__()

        self._n_channels = 2
        if log:
            self.__tb_sub = datetime.now().strftime("%m-%d-%Y_%H%M%S")
            self._tb_path = f"runs/{tag}/{self.__tb_sub}"
            self._writer = SummaryWriter(self._tb_path)
        self._sample_position = 0

        # check for gpu
        self.__device = "cpu"
        if torch.cuda.is_available():
            self.__device_name = torch.cuda.get_device_name(0)
            print(f"GPU acceleration available on {self.__device_name}")

        self._loss_fn = None
        self._optim = None
        self._scheduler = None
        self._cache = None

    @property
    def log_path(self) -> str:
        return self._tb_path

    @log_path.setter
    def log_path(self, data: str):
        self._tb_path = data
    
    @property
    def device(self) -> str:
        return self.__device

    def use_device(self, device: str) -> None:
        self.__device = device
        self._cache = self._cache.to(self.device)
        self.to(self.device)

    def save_to_default(self) -> None:
        device = self.device
        self.use_device("cpu")

        model_tag = datetime.now().strftime("%H%M%S")
        params = self.state_dict()
        torch.save(params, f"{self._tb_path}/model_{model_tag}.torch")

        self.use_device(device)
    
    def load(self, path) -> None:
        raise NotImplementedError()

    def forward(self, x):
        """
        This method performs the forward call on the neural network 
        architecture.

        Args:
            x (Any): The input passed to the defined neural network.

        Raises:
            NotImplementedError: The Base model has not implementation 
                                 for this.
        """
        raise NotImplementedError

    def train_on(self, dataloader: DataLoader, epochs: int = 1, save_every: int = None):
        """
        This method is used to train the model based on the given input
        data.

        Args:
            midi_iterator (Any): The iterator, iterating over each 
                                 midi note and returning its value per 
                                 sample respectively.
            sample_list (List): The expected output given the midi as an input.
        """
        assert self._loss_fn != None
        assert self._optim != None

        total_iters = epochs * len(dataloader)
        p_bar = None

        self.train()
        for e in range(epochs):
            for X, y in dataloader:
                X_midi, X_wave = X[0].to(self.device), X[1].to(self.device)
                y = y.to(self.device)

                # train a batch
                self._optim.zero_grad()
                pred_y = self((X_midi, X_wave))
                loss = self._loss_fn(pred_y, y)
                loss.backward()
                self._optim.step()

                # log for the statistics
                self._writer.add_scalar("Train/loss", loss.item(), self._sample_position)
                self._sample_position += len(X[0])

                # print some gui
                if not p_bar:
                    total_iters *= len(X[0])
                    p_bar = tqdm(total=total_iters)

                p_bar.update(len(X[0]))
                self._writer.flush()
            
            if save_every and e % save_every == 0:
                self.save_to_default()
            
            if self._scheduler:
                self._scheduler.step()
                lr = self._scheduler.get_last_lr()[0]
                self._writer.add_scalar("Train/learning_rate", lr, e)

        self.eval()
        self._writer.flush()

    def predict(self, midi) -> List:
        midi = midi.to(self.device)

        with torch.no_grad():
            _cache = torch.unsqueeze(self._cache, 0)
            sample = self((midi, _cache))
            sequence = len(sample)

            self._cache = torch.roll(self._cache, -sequence)
            self._cache[-sequence, :] = sample

        return sample.detach().cpu().numpy()
