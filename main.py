import numpy as np
import pandas as pd
import scipy.io.wavfile
import torch
from data.midi_set_v2 import MidiWaveDataset
from torch.utils.data import DataLoader
from multiprocessing import freeze_support
from tqdm import tqdm
from model.base import BaseModel
from utils.audio import plot_waveform

from utils.config import config


if __name__ == '__main__':
    freeze_support()

    model: BaseModel = config.get_model("MlpModel")
    model = model(input_size=21, sequence_length=128)
    
    dataset = MidiWaveDataset(
        dataframe=pd.read_hdf("./dataset/train_0/dataset_0.pandas", "midi_wave"),
        device="cpu",
        precision=np.float32
        )

    trainloader = DataLoader(dataset, batch_size=1000, num_workers=2, shuffle=True)
    for data in tqdm(trainloader):
        X, y = data
        model.learn(X, y, epochs=5)
    
    model.save_to_default()
     
    # predict the waveform
    model.reset_cache()
    wave = []
    for i in tqdm(range(1, 44100 * 3)):
        X, y = dataset[i]
        midi, _ = X
        midi = torch.tensor(midi)
        midi = torch.unsqueeze(midi, 0)
        sample = model.predict(midi)
        wave.append(sample)

    wave = np.array(wave).T
    print(wave, wave.shape)
    scipy.io.wavfile.write(f"test.wav", dataset.sample_rate, wave.T)

    plot_waveform(wave, dataset.sample_rate)