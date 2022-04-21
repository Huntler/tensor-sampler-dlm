
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class MidiWaveDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, device: str = "cpu",
                 precision: np.dtype = np.float16) -> None:
                 self._device = device
                 self._precision = precision
                 self._df = dataframe
    
    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, index):
        Xnotes = np.array(self._df["Xnotes"].iloc[index], dtype=self._precision)
        Xsamples = np.array(self._df["Xsamples"].iloc[index], dtype=self._precision)
        
        y = np.array(self._df["y"].iloc[index], dtype=self._precision)
        return (Xnotes, Xsamples), y


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from multiprocessing import freeze_support
    from tqdm import tqdm
    
    freeze_support()
    
    dataset = MidiWaveDataset(
        dataframe=pd.read_hdf("./dataset/train_0/dataset_0.pandas", "midi_wave"),
        device="cpu",
        precision=np.float16
        )

    trainloader = DataLoader(dataset, batch_size=200, num_workers=4, shuffle=True)
    for data in tqdm(trainloader):
        X, y = data