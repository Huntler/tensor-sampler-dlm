from multiprocessing import freeze_support
import torch
import numpy as np
from tqdm import tqdm
from data.midi import MidiWaveDataset
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity


if __name__ == '__main__':
    freeze_support()
    
    dataset = MidiWaveDataset(
        root_dir="./dataset/train_0", 
        dimension=21, 
        note_offset=50,
        sequence_length=500,
        device="cpu",
        precision=torch.float16
        )

    trainloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=True)
    for data in tqdm(trainloader):
        X, y = data
