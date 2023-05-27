from typing import Tuple
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torchaudio
from mido import MidiFile

# dataset located in:
#
# "data"                    <-- folder containg this file
#   "dataset"               <-- not contained in git
#       e.g.: "train_0"
#           "input.mid"
#           "output.wav"

MAX_DATASET_SAMPLE_SIZE = 750_000


class AudioDataset(Dataset):
    def __init__(self, name: str, prev_samples: int, future_samples: int = 1,
                 dimension: int = 90, note_offset: int = 0, normalize: bool = True, 
                 precision: np.dtype = np.float32) -> None:
        # define dataset's parameters
        self.__n_prev = prev_samples
        self.__n_future = future_samples
        self._dimension = dimension
        self._note_offset = note_offset
        self.__precision = precision
        self.__root_dir = f"data/dataset/{name}"
        
        # read the waveform and get the sample rate
        self._metadata = torchaudio.info(f"{self.__root_dir}/output.wav")
        self._wave, self.__sample_rate = torchaudio.load(f"{self.__root_dir}/output.wav")
        self._wave = np.array(self._wave.T.numpy(), dtype=self.__precision)
        assert len(self._wave) <= MAX_DATASET_SAMPLE_SIZE, f"Dataset too big with {len(self._wave)} samples."

        # normalize dataset
        self._normalizer = None
        if normalize:
            self._normalizer = MinMaxScaler()
            self._wave = self._normalizer.fit_transform(self._wave)
            print(f"Dataset boundaries: [{self._normalizer.data_min_};  {self._normalizer.data_max_}]")

    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def channels(self) -> int:
        return self._wave.shape[-1]
    
    @property
    def sequence_length(self) -> int:
        return self.__n_prev + self.__n_future
    
    @property
    def sample_rate(self) -> int:
        return self.__sample_rate
    
    def convert(self, X: np.array) -> np.array:
        if self._normalizer is None:
            return X
        
        return self._normalizer.inverse_transform(X)

    def __len__(self) -> int:
        # there are no samples outside the waveform
        return len(self._wave) - (self.__n_future + self.__n_prev)
        
    def __getitem__(self, index) -> Tuple[Tuple[np.array, np.array], np.array]:
        """Returns midi, wave and label in the form:
            - midi notes from "prev_samples" to "future_samples"
            - waveform from "prev_samples" to "index"
            - label: waveform sample at position "index" (inbetween "prev_samples" and "future_samples")

            The output waveform is considered as cache of previous predictions, the midi range of "index"
            to "future_samples" is used to create some kind of look ahead buffer. 

        Args:
            index (_type_): The current index of the dataset

        Returns:
            Tuple[np.array, np.array]: The samples wave and label.
        """
        # example, n_prev=5 and n_future=3:
        #   wave:   [t-5, t-4, t-3, t-2, t-1, t, t+1, t+2]
        #   sample: [t]

        # slow, since slicing creates a new array
        wave = self._wave[index:index + self.__n_prev + self.__n_future]
        y = self._wave[index + self.__n_prev]

        return wave, y


# test the dataset
# should be executable from the root directory
if __name__ == "__main__":
    from tqdm import tqdm
    dataset = AudioDataset(
        name="train_0",
        prev_samples=4,
        future_samples=2,
    )

    for X, y in tqdm(dataset):

        print(np.argmax(X, axis=-1))
        print(X)
        print(y)
        input("Enter to continue >")