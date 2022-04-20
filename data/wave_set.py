from typing import Tuple
import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from mido import MidiFile


class WaveDataset(Dataset):
    def __init__(self, root_dir: str, sequence_length: int = 64, device: str = "cpu",
                 precision: torch.dtype = torch.float16) -> None:
        """Dataset of MIDI files with corresponding WAVE form.

        Args:
            root_dir (str): Path of the dataset.
        """
        self._device = device
        self._precision = precision
        self._sqeuence_length = sequence_length
        self._root_dir = root_dir

        # read the wave form
        self._metadata = torchaudio.info(f"{root_dir}/output.wav")
        self._wave, self._sample_rate = torchaudio.load(
            root_dir + "/output.wav")
        self._wave = self._wave.T.numpy()
        assert self._metadata.num_frames == len(self._wave)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def __len__(self) -> int:
        return len(self._wave) - self._sqeuence_length
    
    def __getitem__(self, index) -> Tuple[np.array, np.array]:
        X = self._wave[index:index + self._sqeuence_length, :]
        y = self._wave[index + self._sqeuence_length]
        return X, y

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from multiprocessing import freeze_support
    from tqdm import tqdm
    
    freeze_support()
    
    dataset = WaveDataset(
        root_dir="./dataset/train_0",
        sequence_length=128,
        device="cpu",
        precision=torch.float16
        )

    trainloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
    for data in tqdm(trainloader):
        X, y = data