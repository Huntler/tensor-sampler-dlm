from typing import List, Tuple
import numpy as np
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter


class BaseModel(nn.Module):
    def __init__(self, rolling_window_size: int = 1) -> None:
        """
        Base class of a model. This only includes method-playeholders 
        and general methods for statistics etc.
        """
        super(BaseModel, self).__init__()

        self._n_channels = 2
        self._rws = rolling_window_size
        if not self._writer:
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

    def __create_sample_window(self) -> torch.tensor:
        """
        This method created the window holder in which a sequence of samples
        were stored. This is used as an input for training, so this sequence is used
        to calculated the waveform output.

        Returns:
            torch.tensor: The window.
        """
        if self._rws == 1:
            sample_window = np.expand_dims(
                np.zeros((self._n_channels,), dtype=np.float32))
        else:
            sample_window = np.repeat(
                [np.zeros((self._n_channels,), dtype=np.float32)], self._rws, axis=0)
        
        return torch.from_numpy(sample_window)

    def __roll_window(self, window: torch.tensor, to_append: torch.tensor) -> torch.tensor:
        """
        This method rolls a given window by one element and filles the given 
        element to the empty slot at the end.

        Args:
            window (torch.tensor): The window which is going to be rolled.
            to_append (torch.tensor): The element inserted at the free slot.

        Returns:
            Tuple: A tensor of the updated window.
        """
        window = torch.cat((window[1:], torch.unsqueeze(to_append, 0)), 0)
    
    def __to_tensors(self, window: torch.tensor, midi: List, sample: List) -> Tuple:
        """This method creates the tensors out of our input and expected output data. Also, this 
        method flattens and concatenates the window and midi input.

        Args:
            window (torch.tensor): The window containing the previous predicted samples.
            midi (List): The current notes to play.
            sample (List): The sample which is expected.

        Returns:
            Tuple: A tuple contain X and y as tensors.
        """

        torch_sample_window = torch.flatten(window)
        torch_midi = torch.from_numpy(midi)

        X = torch.cat((torch_midi, torch_sample_window), 0).float()
        y = torch.from_numpy(sample).float()

        return X, y

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
            sample_window = self.__create_sample_window()

            # run over all midi samples generated and connect those with
            # the corresponding wave form sample
            for i, midi in enumerate(midi_iterator):
                # convert the the window and sample to a torch so it can
                # be passed into our model. Also, consider the device it is
                # running on
                X, y = self.__to_tensors(sample_window, midi, sample_list[i])

                # perform the presiction and measure the loss between the prediction
                # and the expected output
                pred_y = self(X)

                # roll the window to get a free slot and insert the new sample calculated
                self.__roll_window(sample_window, pred_y)

                # calculate the gradient using backpropagation of the loss
                loss = self._loss_fn(pred_y, y)
                self._optim.zero_grad
                loss.backward()
                self._optim.step()

                # log for the statistics
                self._writer.add_scalar("Train/loss", loss, i)

        self.eval()
        self._writer.flush()
    
    def test(self, midi_iterator, sample_list):
        """
        This method is used to test a trained model. Also, this method logs the
        predicted and expected waveform to the tensorboard, so both can be compared 
        manually.

        Args:
            midi_iterator (Any): The iterator for all midi notes.
            sample_list (List): The waveform as a python list.
        """
        sample_list = np.asarray(sample_list, dtype=np.float32)

        # define the amount of midi message we are looking at when predicting the
        # upcomming wave form sample
        sample_window = self.__create_sample_window()

        # run over all midi samples generated and connect those with
        # the corresponding wave form sample
        for i, midi in enumerate(midi_iterator):
            # convert the the window and sample to a torch so it can
            # be passed into our model. Also, consider the device it is
            # running on
            X, y = self.__to_tensors(sample_window, midi, sample_list[i])

            # perform the presiction and measure the loss between the prediction
            # and the expected output
            pred_y = self(X)
            loss = self._loss_fn(pred_y, y)

            # roll the window to get a free slot and insert the new midi message
            self.__roll_window(sample_window, midi)

            # log for the statistics
            self._writer.add_scalar("Test/loss", loss, i)

            # plot the predicted and expected waveforms
            self._writer.add_scalar("Test/predicted_left", pred_y[0], i)
            self._writer.add_scalar("Test/expected_left", y[0], i)
        
        self._writer.flush()
