import numpy as np
import scipy.io
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
        root_dir="./dataset/train_0", 
        dimension=21, 
        note_offset=50,
        sequence_length=128,
        device="cpu",
        precision=np.float32
        )

    trainloader = DataLoader(dataset, batch_size=16, num_workers=2, shuffle=True)
    i = 0
    for data in tqdm(trainloader):
        X, y = data
        model.learn(X, y)
        i += 1
        if i >= 500:
            break
    
    model.save_to_default()
     
    # predict the waveform
    model.reset_cache()
    wave = []
    for i in tqdm(range(1, 44100 * 3)):
        X, y = dataset[i]
        midi, _ = X
        sample = model.predict(torch.tensor(midi))
        wave.append(sample)

    wave = np.array(wave).T
    print(wave, wave.shape)
    scipy.io.wavfile.write(f"test.wav", dataset.sample_rate, wave.T)

    plot_waveform(wave, dataset.sample_rate)