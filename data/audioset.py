from typing import Tuple
import numpy as np
from torch.utils.data import Dataset
import torchaudio

# dataset located in:
#
# "data"                    <-- folder containg this file
#   "dataset"               <-- not contained in git
#       e.g.: "train_0"
#           "input.mid"
#           "output.wav"

class AudioDataset(Dataset):
    def __init__(self, name: str, prev_samples: int, future_samples: int = 1,
                 precision: np.dtype = np.float32) -> None:
        # define dataset's parameters
        self.__n_prev = prev_samples
        self.__n_future = future_samples
        self.__precision = precision
        self.__root_dir = f"data/dataset/{name}"
        
        # read the waveform and get the sample rate
        self._metadata = torchaudio.info(f"{self.__root_dir}/output.wav")
        self.__wave, self.__sample_rate = torchaudio.load(f"{self.__root_dir}/output.wav")
        self.__wave = np.array(self.__wave.T.numpy(), dtype=self.__precision)

        # read the midi file
        # TODO: read the midi file such that an arbitary index of it can be returned later on

    @property
    def sample_rate(self) -> int:
        return self.__sample_rate

    def __len__(self) -> int:
        # there are no samples outside the waveform
        return len(self.__wave) - self.__n_future
        
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
            Tuple[Tuple[np.array, np.array], np.array]: The samples (midi, wave) and label.
        """
        # TODO: build the structure explained in the documentation
        return (None, None), None


# test the dataset
# should be executable from the root directory
if __name__ == "__main__":
    dataset = AudioDataset(
        name="train_0",
        prev_samples=1024,
        future_samples=512,
    )

    for X, y in dataset:
        X_midi, X_wave = X
        print(X.shape, y.shape)