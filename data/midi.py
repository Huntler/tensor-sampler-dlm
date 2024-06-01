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
                 input_sequence: int = 64, output_sequence: int = 1, precision: np.dtype = np.float16) -> None:
        """Dataset of MIDI files with corresponding WAVE form.

        Args:
            root_dir (str): Path of the dataset.
        """
        self._precision = precision
        self._dimension = dimension
        self._note_offset = note_offset
        self._sqeuence_length = input_sequence
        self._out_seq_length = output_sequence
        self._root_dir = f"data/dataset/{name}"

        # read the wave form
        self._metadata = torchaudio.info(f"{self._root_dir}/output.wav", backend="ffmpeg")
        self._wave, self._sample_rate = torchaudio.load(
            self._root_dir + "/output.wav", backend="ffmpeg")
        self._wave = self._wave.T.numpy()
        self._wave = np.array(self._wave, dtype=self._precision)
        assert self._metadata.num_frames == len(self._wave)

        # normalize dataset
        minmax = MinMaxScaler()
        self._wave = minmax.fit_transform(self._wave)
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

        # create sequence holder
        self._midi_seq = np.zeros((self._sqeuence_length, self._dimension))
        self._wave_seq = np.zeros((self._sqeuence_length, self._metadata.num_channels))

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
        return self._total_time - self._sqeuence_length
    
    def __getitem__(self, index) -> Tuple[np.array, np.array]:
        # speed up the process
        end_index = index + self._sqeuence_length
        seq_time_index = 0

        # generate the sequences
        midi_seq = np.zeros((self._sqeuence_length, self._dimension), dtype=self._precision)
        wave_seq = np.zeros((self._sqeuence_length, 2), dtype=self._precision)
        _tmp = self._wave[index:end_index, :]
        wave_seq[:len(_tmp)] = _tmp

        to_pred = np.zeros((self._out_seq_length, 2), dtype=self._precision)
        _tmp = self._wave[end_index:end_index + self._out_seq_length, :]
        to_pred[:len(_tmp)] = _tmp

        # get the sequence
        for seq_index in range(index, end_index):
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
        
        return (midi_seq, wave_seq), to_pred


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from multiprocessing import freeze_support
    from tqdm import tqdm
    
    freeze_support()

    dataset = MidiWaveDataset(
        name="train_0",
        dimension=65,
        input_sequence=1024,
        output_sequence=256
        )

    trainloader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=True)
    for data in tqdm(trainloader):
        X, y = data