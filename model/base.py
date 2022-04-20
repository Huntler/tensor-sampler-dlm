import contextlib
from datetime import datetime
from typing import List, Tuple
import numpy as np
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter


@contextlib.contextmanager
def no_autocast():
    yield None


class BaseModel(nn.Module):
    def __init__(self) -> None:
        """
        Base class of a model. This only includes method-playeholders 
        and general methods for statistics etc.
        """
        super(BaseModel, self).__init__()

        self._n_channels = 2
        if self._writer is None:
            self.__tb_sub = datetime.now().strftime("%m-%d-%Y_%H%M%S")
            self._tb_path = f"runs/{self.__tb_sub}"
            self._writer = SummaryWriter(self._tb_path)
        self.__sample_position = 0

        # check for gpu
        self._device = "cpu"
        if torch.cuda.is_available():
            self.__device_name = torch.cuda.get_device_name(0)
            print(f"GPU acceleration available on {self.__device_name}")

        self._loss_fn = None
        self._optim = None

        # for prediction
        self._cache = None

    @property
    def log_path(self) -> str:
        return self._tb_path

    def use_device(self, device: str) -> None:
        self._device = device
        self.to(self._device)

    def save_to_default(self) -> None:
        model_tag = datetime.now().strftime("%H%M%S")
        params = self.state_dict()
        torch.save(params, f"{self._tb_path}/model_{model_tag}.torch")
    
    def load(self, path) -> None:
        raise NotImplementedError()

    def reset_cache(self) -> None:
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

    def learn(self, X, y, epochs: int = 1):
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

        # measure history
        losses = []

        for e in range(0, epochs):
            with torch.cuda.amp.autocast() if self._device == "cuda" else no_autocast():
                # perform the presiction and measure the loss between the prediction
                # and the expected output
                pred_y = self(X)

                # calculate the gradient using backpropagation of the loss
                loss = self._loss_fn(pred_y, y)

            self._optim.zero_grad()
            loss.backward()
            self._optim.step()

            losses.append(loss.item())

        # log for the statistics
        losses = np.mean(losses, axis=0)
        self._writer.add_scalar("Train/loss", loss, self.__sample_position)
        self.__sample_position += len(X)

        self.eval()
        self._writer.flush()

    def predict(self, midi) -> List:
        assert self._cache != None

        with torch.no_grad():
            sample = self((midi, self._cache))

            self._cache = torch.roll(self._cache, -1)
            self._cache[-1, :] = sample

        return sample.numpy()
