from typing import Tuple
import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from mido import MidiFile
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# @DeprecationWarning
class MidiWaveDataset(Dataset):
    def __init__(self, name: str, dimension: int = 65, note_offset: int = 0,
                 prev_samples: int = 500, future_samples: int = 500, 
                 normalize: bool = True, precision: np.dtype = np.float32) -> None:
        """Dataset of MIDI files with corresponding WAVE form.

        Args:
            root_dir (str): Path of the dataset.
        """
        self._precision = precision
        self._dimension = dimension
        self._note_offset = note_offset
        self.__n_prev = prev_samples
        self.__n_future = future_samples
        self._root_dir = f"data/dataset/{name}"

        # read the wave form
        self._metadata = torchaudio.info(f"{self._root_dir}/output.wav")
        self.__wave, self._sample_rate = torchaudio.load(
            self._root_dir + "/output.wav")
        self.__wave = self.__wave.T.numpy()
        self.__wave = np.array(self.__wave, dtype=self._precision)
        assert self._metadata.num_frames == len(self.__wave)

        # normalize dataset
        if normalize:
            minmax = MinMaxScaler()
            self.__wave = minmax.fit_transform(self.__wave)
            print(f"Dataset boundaries: [{minmax.data_min_};  {minmax.data_max_}]")

        # read the midi file's track
        # create the midi file object and the list in which the notes
        # and timestamp were stored
        self._midi_file = MidiFile(f"{self._root_dir}/input.mid")

        self._midi_track = {}
        total_time = 0
        for msg in self._midi_file:
            # we are only interested in messages containing a note
            if "note" in msg.dict().keys():
                # rise the total playtime
                total_time += int(msg.time * self._sample_rate)
                note_index = msg.note - self._note_offset
                active_notes = self._midi_track.get(total_time, None)

                # if there are no active notes registered for the given time stamp
                if type(active_notes) == type(None):
                    note_value = 0 if msg.type == "note_off" else 1
                    self._midi_track[total_time] = self.__one_hot(
                        note_index, note_value)
                    continue

                note_value = -1 if msg.type == "note_off" else 1
                active_notes += self.__one_hot(note_index, note_value)
                self._midi_track[total_time] = active_notes

        self._total_time = total_time
        self._start_times = [_ for _ in self._midi_track.keys()]
        self._start_times.sort()

    def __one_hot(self, index: int, value: int, tensor: bool = False) -> torch.tensor:
        one_hot = np.zeros((self._dimension), dtype=np.uint8)
        one_hot[index] = value
        return np.array(one_hot, dtype=self._precision)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def __len__(self) -> int:
        return self._total_time - (self.__n_future + self.__n_prev)
    
    def __getitem__(self, midi_start) -> Tuple[np.array, np.array]:
        # example, n_prev=5 and n_future=3:
        #   midi:   [t-5, t-4, t-3, t-2, t-1, t, t+1, t+2]
        #   wave:   [t-1, t-4, t-3, t-2, t-1]
        #   sample: [t]

        # define indices and ranges
        index = midi_start + self.__n_prev
        midi_end = index + self.__n_future
        
        # speed up the process by remembering the last sequence matching the index
        seq_time_index = 0

        # generate the sequences
        midi_seq = np.zeros((self.__n_prev + self.__n_future, self._dimension), dtype=self._precision)

        # (1) wave sequence, stop index is exclusive
        wave_seq = self.__wave[midi_start:index, :]
        sample = self.__wave[index]

        # (2) midi sequence in range "midi_start" to "midi_end"
        for seq_index in range(midi_start, midi_end):
            for time_index, start_time in enumerate(self._start_times[seq_time_index:]):
                time_index += seq_time_index
                # if the start_time of an midi message gets greater than the index, then
                # we will not find one which is lower (array was sorted before)
                if start_time > seq_index:
                    break

                # there is no midi message after this one, we are at the end
                if len(self._start_times) <= time_index:
                    midi_seq[-(index - seq_index), :] = self._midi_track[start_time]
                    seq_time_index = time_index
                    break

                # if the following message's start time is bigger than the index, we can safely
                # add the current message to the sequence
                if self._start_times[time_index + 1] >= seq_index:
                    midi_seq[-(index - seq_index), :] = self._midi_track[start_time]
                    seq_time_index = time_index
                    break
        
        return (midi_seq, wave_seq), sample


if __name__ == "__main__":
    from tqdm import tqdm
    
    n_prev = 500
    n_future = 500

    dataset = MidiWaveDataset(
        name="train_0",
        dimension=65,
        prev_samples=n_prev,
        future_samples=n_future
        )

    for data in tqdm(dataset):
        X, y = data
        X_midi, X_wave = X

        assert len(X_midi) == n_prev + n_future, f"length was {len(X_midi)} but expected {n_prev + n_future}"
        assert len(X_wave) == n_prev, f"length was {len(X_wave)} but expected {n_prev}"