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

        # create logger instance (TensorBoard) and set position of pointer
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

        # define variables for loss function, optimizer and scheduler
        self._loss_fn = None
        self._optim = None
        self._scheduler = None

        # define parameters for sound generation, such as output channel size and
        # input buffer from previous wave form
        self._n_channels = 2
        self._cache = None
    
    @property
    def channels(self) -> int:
        return self._n_channels

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
        """Moves the network and cache storage to the specified device.

        Args:
            device (str): Device should be supported by PyTorch e.g. 'cpu', 'cuda', 'mps', ...
        """
        self.__device = device
        self._cache = self._cache.to(self.device)
        self.to(self.device)

    def save_to_default(self) -> None:
        """Moves the network back to CPU and saves it to disk. Then the network is moved back 
        to its original device to be able to further train it.
        """
        # remember current device and move network
        device = self.device
        self.use_device("cpu")

        # create a unique model tag and save it
        model_tag = datetime.now().strftime("%H%M%S")
        params = self.state_dict()
        torch.save(params, f"{self._tb_path}/model_{model_tag}.torch")

        # finally move the network back
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
        # loss function and optimizers are needed
        assert self._loss_fn != None
        assert self._optim != None

        # create tge progress bar variable and calculate the amount of samples to train
        total_iters = epochs * len(dataloader)
        p_bar = None

        # start the training loop
        self.train()
        for e in range(epochs):
            for X, y in dataloader:
                # move samples/batch to train onto the specified device
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

                # print the progress bar showing the trainings progress
                if not p_bar:
                    total_iters *= len(X[0])
                    p_bar = tqdm(total=total_iters)

                p_bar.update(len(X[0]))
                self._writer.flush()
            
            # save the model every x epochs if wanted, value has to be above 0
            if save_every and e % save_every == 0:
                self.save_to_default()
            
            # if a scheduler was set, then use it to adapt the learning rate
            if self._scheduler:
                self._scheduler.step()
                lr = self._scheduler.get_last_lr()[0]
                self._writer.add_scalar("Train/learning_rate", lr, e)

        self.eval()
        self._writer.flush()

    def predict(self, midi) -> List:
        # move the input onto the same device as the model is on
        midi = midi.to(self.device)

        # calculating a gradient is not needed, disable that to improve performance
        with torch.no_grad():
            # send the cache and midi input to the network
            _cache = torch.unsqueeze(self._cache, 0)
            sample = self((midi, _cache))
            sequence = len(sample)

            # update the cache based on the prediction
            self._cache = torch.roll(self._cache, -sequence)
            self._cache[-sequence, :] = sample

        # always move the sample to cpu and return it
        return sample.detach().cpu().numpy()
