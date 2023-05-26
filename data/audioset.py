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
        self.__wave, self.__sample_rate = torchaudio.load(f"{self.__root_dir}/output.wav")
        self.__wave = np.array(self.__wave.T.numpy(), dtype=self.__precision)
        assert len(self.__wave) <= MAX_DATASET_SAMPLE_SIZE, f"Dataset too big with {len(self.__wave)} samples."

        # normalize dataset
        if normalize:
            minmax = MinMaxScaler()
            self.__wave = minmax.fit_transform(self.__wave)
            print(f"Dataset boundaries: [{minmax.data_min_};  {minmax.data_max_}]")

        # read the midi file such that an arbitary index of it can be returned later on
        self._midi_file = MidiFile(f"{self.__root_dir}/input.mid")

        # store the midi notes as hot-encoded vector of booleans
        self._midi = np.zeros((len(self.__wave), self._dimension), dtype=np.bool_)

        # calculate max time of midi file
        max_time = 0
        for msg in self._midi_file.tracks[-1]:
            if "note" in msg.dict().keys():
                max_time += msg.time
        time_factor = self._metadata.num_frames / max_time

        # create array
        for i, msg in enumerate(self._midi_file.tracks[-1]):
            # we are only interested in messages containing a note
            if "note_off" in msg.type:
                note_index = msg.note - self._note_offset

                # we found the note_off command, now add this note to the array until we find the start
                for j in range(i, 0, -1):
                    # self._midi[j] += self.__one_hot(note_index, 1)
                    old_msg = self._midi_file.tracks[-1][j]
                    if old_msg.type != "note_on":
                        continue

                    # start of command found, add the one-hot encoded vector old_msg.time times
                    if old_msg.note - self._note_offset == note_index:
                        for sample in range(0, msg.time):
                            self._midi[j + sample] += self.__one_hot(note_index, 1)

    def __one_hot(self, index: int, value: int, tensor: bool = False) -> torch.tensor:
        one_hot = np.zeros((self._dimension), dtype=np.uint8)
        one_hot[index] = value
        return np.array(one_hot, dtype=np.bool_)

    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def sample_rate(self) -> int:
        return self.__sample_rate

    def __len__(self) -> int:
        # there are no samples outside the waveform
        return len(self.__wave) - (self.__n_future + self.__n_prev)
        
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
        # example, n_prev=5 and n_future=3:
        #   midi:   [t-5, t-4, t-3, t-2, t-1, t, t+1, t+2]
        #   wave:   [t-5, t-4, t-3, t-2, t-1]
        #   sample: [t]

        midi = self._midi[index:index + self.__n_prev + self.__n_future]
        wave = self.__wave[index:index + self.__n_prev]
        y = self.__wave[index + self.__n_prev]

        return (np.array(midi, dtype=self.__precision), np.array(wave, dtype=self.__precision)), \
                np.array(y, dtype=self.__precision)


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
        X_midi, X_wave = X

        print(np.argmax(X_midi, axis=-1))
        print(X_wave)
        print(y)
        input("Enter to continue >")