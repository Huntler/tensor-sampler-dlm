from datetime import datetime
from typing import List, Tuple
import numpy as np
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter


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
            self.__tb_path = f"runs/{self.__tb_sub}"
            self._writer = SummaryWriter(self.__tb_path)
        self.__sample_position = 0

        # check for gpu
        self.__device = "cpu"
        if torch.cuda.is_available():
            self.__device_name = torch.cuda.get_device_name(0)
            print(f"GPU acceleration available on {self.__device_name}")
    
    def use_device(self, device: str) -> None:
        self.__device = device
        self.to(self.__device)
    
    def save_to_default(self) -> None:
        model_tag = datetime.now().strftime("%H%M%S")
        torch.save(self, f"{self.__tb_path}/model_{model_tag}.torch")

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

    def learn(self, midi, sample_list, epochs: int = 1):
        """
        This method is used to train the model based on the given input
        data.

        Args:
            midi_iterator (Any): The iterator, iterating over each 
                                 midi note and returning its value per 
                                 sample respectively.
            sample_list (List): The expected output given the midi as an input.
        """
        # measure history
        losses = []

        X = midi
        #FIXME: get batches working
        print(X[0])
        y = sample_list
        for e in range(0, epochs):

            # perform the presiction and measure the loss between the prediction
            # and the expected output
            pred_y = self(X)

            # calculate the gradient using backpropagation of the loss
            loss = self._loss_fn(pred_y, y[i])
            self._optim.zero_grad
            loss.backward()
            self._optim.step()

            losses.append(loss.item())

        # log for the statistics
        losses = np.mean(losses, axis=0)
        for i, loss in enumerate(losses):
            self._writer.add_scalar("Train/loss", loss, self.__sample_position + i)
        self.__sample_position += len(losses)

        self.eval()
        self._writer.flush()
    
    def predict(self, midi_iterator) -> List:
        out = []
        with torch.no_grad():
            for i, X in enumerate(midi_iterator):
                pred_y = self(X)
                out.append(pred_y)
        
        return out
