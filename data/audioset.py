"""This module loads audio files as dataset."""
from typing import Tuple
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import torchaudio


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, d_type: str = "train",
                 normalize: bool = True,
                 bounds: Tuple[int] = (-1, 1),
                 future_steps: int = 11025,
                 sequence_length: int = 11025,
                 dimension: int = 2,
                 precision: np.dtype = np.float32,
                 custom_path: str = None,
                 ae_mode: bool = False) -> None:
        """Loads a dataset from the default path "data/dataset/<d_type>" which should contain
        an "input.wav" and an "output.wav". The audio files require matching sample rate and 
        number of samples. The dataset can be used to train an auto-encoder on the input data only.

        Args:
            d_type (str, optional): The dataset type. Defaults to "train".
            normalize (bool, optional): Normalize the data using MinMax. Defaults to True.
            bounds (Tuple[int], optional): The boundaries for MinMax scaling. Defaults to (-1, 1).
            future_steps (int, optional): The steps to predict. Defaults to 11025.
            sequence_length (int, optional): The steps fed into the model to predict. Defaults 
            to 11025.
            dimension (int, optional): The dimension (channels) of the audio files. Defaults to 2.
            precision (np.dtype, optional): The precision to work with. Defaults to np.float32.
            custom_path (str, optional): Overwrites the default dataset location. Defaults to None.
            ae_mode (bool, optional): Returns the input also as output to train an auto-encoder. 
            Defaults to False.
        """

        # define dataset's parameters
        self._n_prev = sequence_length
        self._n_future = future_steps
        self._dimension = dimension
        self._ae_mode = ae_mode
        self._precision = precision

        path = custom_path if custom_path else "data/dataset"
        self._root_dir = f"{path}/{d_type}"

        # read the input waveform and get the sample rate
        self._wave_in, self._sample_rate = torchaudio.load(
            f"{self._root_dir}/input.wav")
        self._wave_in = np.array(
            self._wave_in.T.numpy(), dtype=self._precision)

        # read the output waveform and get the sample rate
        self._wave_out, sample_rate_out = torchaudio.load(
            f"{self._root_dir}/output.wav")
        self._wave_out = np.array(
            self._wave_out.T.numpy(), dtype=self._precision)

        assert self._wave_in.shape == self._wave_out.shape
        assert self._sample_rate == sample_rate_out

        # normalize dataset
        self._minmax = None
        if normalize:
            self._minmax = MinMaxScaler(bounds)
            self._minmax = self._minmax.fit(self._wave_in)
            self._minmax = self._minmax.fit(self._wave_out)
            self._wave_in = self._minmax.transform(self._wave_in)
            print(
                f"Dataset boundaries: [{self._minmax.data_min_}, {self._minmax.data_max_}]")

    def reverse_scaling(self, data: np.array) -> np.array:
        """Reverses the scaling if the data was normalized.

        Args:
            data (np.array): The data to inverse transform.

        Returns:
            np.array: The scaled back data.
        """
        if not self._minmax:
            return data

        return self._minmax.inverse_transform(data)

    @property
    def dimension(self) -> int:
        """The amount of channels of the loaded audio file.

        Returns:
            int: Audio channels.
        """
        return self._dimension

    @property
    def sample_rate(self) -> int:
        """The sample rate of the loaded audio file.

        Returns:
            int: Sample rate (e.g. 44100).
        """
        return self._sample_rate

    @property
    def sample_size(self) -> int:
        """Returns the sample size of the dataset.

        Returns:
            int: The dataset's sample size.
        """
        return len(self._wave_in)

    def __len__(self) -> int:
        # there are no samples outside the waveform
        return len(self._wave_in) - (self._n_future + self._n_prev)

    def __getitem__(self, index) -> Tuple[np.array, np.array]:
        wave_in = self._wave_in[index:index + self._n_prev]
        if not self._ae_mode:
            wave_out = self._wave_out[index + self._n_prev:index +
                                      self._n_prev + self._n_future]

            return wave_in, wave_out
        else:
            return wave_in, wave_in
