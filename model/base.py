import numpy as np
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter

from data.iterators import MAX_N_NOTES


class BaseModel(nn.Module):
    def __init__(self, rolling_window_size: int = 1) -> None:
        """
        Base class of a model. This only includes method-playeholders 
        and general methods for statistics etc.
        """
        super(BaseModel, self).__init__()

        self._rws = rolling_window_size
        self._writer = SummaryWriter()

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

    def learn(self, midi_iterator, sample_list, epochs: int = 1):
        """
        This method is used to train the model based on the given input
        data.

        Args:
            midi_iterator (Any): The iterator, iterating over each 
                                 midi note and returning its value per 
                                 sample respectively.
            sample_list (List): The expected output given the midi as an input.
        """
        sample_list = np.asarray(sample_list, dtype=np.float32)
        for e in range(0, epochs):
            # define the amount of midi message we are looking at when predicting the
            # upcomming wave form sample
            if self._rws == 1:
                midi_sample_window = np.expand_dims(
                    np.zeros((MAX_N_NOTES,), dtype=np.uint8))
            else:
                midi_sample_window = np.repeat(
                    [np.zeros((MAX_N_NOTES,), dtype=np.uint8)], self._rws, axis=0)

            # run over all midi samples generated and connect those with
            # the corresponding wave form sample
            for i, midi in enumerate(midi_iterator):
                # roll the window to get a free slot and insert the new midi message
                midi_sample_window[:-1] = midi_sample_window[1:]
                midi_sample_window[-1] = midi

                # convert the the window and sample to a torch so it can
                # be passed into our model. Also, consider the device it is
                # running on
                X = torch.from_numpy(midi_sample_window).float()
                y = torch.from_numpy(sample_list[i]).float()

                # perform the presiction and measure the loss between the prediction
                # and the expected output
                pred_y = self(X)
                loss = self._loss_fn(pred_y, y)

                # calculate the gradient using backpropagation of the loss
                self._optim.zero_grad
                loss.backward()
                self._optim.step()

                # log for the statistics
                self._writer.add_scalar("Loss/train", loss, i)

        self.eval()
