import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np

from utils.audio import print_stats
from .loading import load_midi_file


class MidiWaveDataset(Dataset):
    def __init__(self, root_dir: str, num_notes=20) -> None:
        """Dataset of MIDI files with corresponding WAVE form.

        Args:
            root_dir (str): Path of the dataset.
        """
        self.__root_dir = root_dir
        self.__midi_file, _ = load_midi_file(root_dir + "/input.mid")
        self.__metadata = torchaudio.info(root_dir + "/output.wav")

        self.__wave, self.__sample_rate = torchaudio.load(root_dir + "/output.wav")
        self.__wave = self.__wave.T

        print("Expected stats:")
        print_stats(self.__wave, self.__sample_rate)

        self.__active_playing = torch.tensor(np.zeros((num_notes,), dtype=np.int))
        # self.__active_playing = np.array([[torch.tensor(0)] for _ in range(num_notes)])
        assert self.__metadata.num_frames == len(self.__wave)
    
    @property
    def sample_rate(self) -> int:
        return self.__metadata.sample_rate
    
    def __len__(self) -> int:
        return self.__metadata.num_frames

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # only lower the times of notes that are played
        # not the fastest but effective
        self.__active_playing[self.__active_playing > 0] -= 1
        
        # get notes, if their start time matches the current time
        notes = self.__midi_file.get(idx, [])

        # then iterate of the notes and add those to the active playing array
        for delta_time, note_index, start_time in notes:
            self.__active_playing[note_index] = delta_time
        
        return self.__active_playing.float(), self.__wave[idx]
