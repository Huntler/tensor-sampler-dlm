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
                active_notes = self._midi_track.get(total_time, -1)

                # if there are no active notes registered for the given time stamp
                if active_notes == -1:
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
        return self._total_time
    
    def __getitem__(self, index) -> Tuple[np.array, np.array]:
        # generate the sequences
        midi_seq = np.zeros((self._sqeuence_length, self._dimension))
        wave_seq = self._wave[index]

        # speed up the process
        start_index = index - self._sqeuence_length
        start_index = 0 if start_index < 0 else start_index
        seq_time_index = 0

        # get the sequence
        for seq_index in range(start_index, index):
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
                
        return midi_seq, wave_seq

