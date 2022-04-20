from typing import Tuple
import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from mido import MidiFile


class MidiWaveDataset(Dataset):
    def __init__(self, root_dir: str, dimension: int = 20, note_offset: int = 0,
                 sequence_length: int = 64, device: str = "cpu",
                 precision: torch.dtype = torch.float16) -> None:
        """Dataset of MIDI files with corresponding WAVE form.

        Args:
            root_dir (str): Path of the dataset.
        """
        self._device = device
        self._precision = precision
        self._dimension = dimension
        self._note_offset = note_offset
        self._sqeuence_length = sequence_length
        self._root_dir = root_dir

        # read the wave form
        self._metadata = torchaudio.info(f"{root_dir}/output.wav")
        self._wave, self._sample_rate = torchaudio.load(
            root_dir + "/output.wav")
        self._wave = self._wave.T.numpy()
        assert self._metadata.num_frames == len(self._wave)

        # read the midi file's track
        # create the midi file object and the list in which the notes
        # and timestamp were stored
        self._midi_file = MidiFile(f"{root_dir}/input.mid")

        self._midi_track = {}
        total_time = 0
        for msg in self._midi_file:
            # we are only interested in messages containing a note
            if "note" in msg.dict().keys():
                # rise the total playtime
                total_time += int(msg.time * self._sample_rate)
                note_index = msg.note - self._note_offset
                active_notes = self._midi_track.get(total_time, [])

                # if there are no active notes registered for the given time stamp
                if active_notes == []:
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

        if tensor:
            return torch.tensor(one_hot, dtype=self._precision, device=self._device)
        else:
            return one_hot

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def __len__(self) -> int:
        return self._total_time - self._sqeuence_length
    
    def __getitem__(self, index) -> Tuple[np.array, np.array]:
        # get the end index of our sequence
        end_index = index + self._sqeuence_length

        # generate the sequences
        midi_seq = None
        wave_seq = self._wave[index:end_index, :]
        wave_sample = self._wave[index]

        # get the end index of our sequence
        end_index = index + self._sqeuence_length

        # find the start time index to get the correct notes for the current index
        for time_index, start_time in enumerate(self._start_times):
            # at the end of our midi file, add all notes to our midi_seq
            if len(self._start_times) <= time_index:
                # make sure to stay below the dataset's length
                end_index = min(self._total_time, end_index)
                midi_seq = np.repeat([self._midi_track[start_time]], self._sqeuence_length, axis=0)
                wave_seq = self._wave[index:end_index, :]
                break

            if self._start_times[time_index + 1] < index:
                continue

            # if the sequence is fully contained in the start_time of our note, 
            # then simply add it
            if index > start_time and end_index < self._start_times[time_index + 1]:
                midi_seq = np.repeat([self._midi_track[start_time]], self._sqeuence_length, axis=0)
                break

            # in this case, our output sequence includes more than one start_time
            if index > start_time and end_index > self._start_times[time_index + 1]:
                # create the first part of our sequence containing notes from the current 
                # start_time
                first_seq_length = self._start_times[time_index + 1] - index
                midi_seq = np.repeat([self._midi_track[start_time]], first_seq_length, axis=0)

                # find and create all following sub-sequences until end_index < start_time
                i = 1
                while end_index > self._start_times[time_index + i]:
                    # get the sub_sequence length first, which is either the time until the next 
                    # note gets played or the remaining sequence length
                    sub_seq_length = self._start_times[time_index + i] - self._start_times[time_index]
                    sub_seq_length = min(self._sqeuence_length - first_seq_length, sub_seq_length)

                    sub_seq = np.repeat([self._midi_track[start_time]], sub_seq_length, axis=0)
                    midi_seq = np.vstack((midi_seq, sub_seq))
                    i += 1

                break

        X = (np.array(midi_seq, dtype=self._precision), np.array(wave_seq, self._precision))
        return X, wave_sample



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from multiprocessing import freeze_support
    from tqdm import tqdm
    
    freeze_support()
    
    dataset = MidiWaveDataset(
        root_dir="./dataset/train_1", 
        dimension=21, 
        note_offset=50,
        sequence_length=128,
        device="cpu",
        precision=torch.float16
        )

    trainloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
    for data in tqdm(trainloader):
        X, y = data