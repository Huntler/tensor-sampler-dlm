import argparse
import numpy as np

import torch
from data.dataset import MidiWaveDataset
from model.crispy_cranberry import CrispyCranberry
from model.little_lion import LittleLion
from utils.audio import play_audio, plot_specgram, plot_waveform, print_stats
from model.wispy_waterfall import WispyWaterfall
from torch.utils.data import DataLoader
from multiprocessing.spawn import freeze_support
from tqdm import tqdm

# FIXME: implement batch normalization (Lecture 2, Slide 102-103)


def train_mode():
    # define parameters
    model_name = LittleLion
    dataset_name = "train_0"
    batch_size_in_seconds = 1
    device = "cpu"
    precision = torch.float16 if device == "cuda" else torch.float32
    epochs = 1

    freeze_support()

    # create the dataset loader
    dataset = MidiWaveDataset(root_dir=f"dataset/{dataset_name}", precision=precision)
    batch_size = int(dataset.sample_rate * batch_size_in_seconds)  # seconds

    # FIXME: Normalize using transformers
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    print(f"Training on {int(len(dataset) / batch_size)} batches, each {batch_size} samples " +
          f"({batch_size / dataset.sample_rate} sec) big.")

    # create the DLM to use
    model = model_name(precision=precision)
    model.use_device(device)

    # train the model
    for epoch in range(epochs):
        for notes_active, wave_sample in tqdm(dataloader):
            notes_active, wave_sample = notes_active.to(
                device), wave_sample.to(device)
            model.learn(notes_active, wave_sample, epochs=3)

    model.save_to_default()


def load_mode(path):
    # load the model given the path
    model = CrispyCranberry(log=False)
    model.load(path)

    # create the test dataset and execute the model on it
    dataset = MidiWaveDataset(root_dir="dataset/train_1")
    batch_size = int(dataset.sample_rate * 1)  # seconds
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=8)

    # predict the waveform
    wave = []
    for notes_active, _ in tqdm(dataloader):
        window = model.predict(notes_active)
        for sample in window:
            wave.append(sample.numpy())

    wave = np.array(wave).T
    print_stats(wave, dataset.sample_rate)
    plot_waveform(wave, dataset.sample_rate)
    play_audio(wave, dataset.sample_rate)

    import scipy.io.wavfile
    scipy.io.wavfile.write("train_1.wav", dataset.sample_rate, wave.T)


# how to use the dataset MidiWave dataset
# FIXME: remove data/iterators.py afterwards
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or load a VST Tensor-Sample DLM.")
    parser.add_argument("--train", action='store_true', default=False,
                        dest="train", help="Enables the training mode.")
    parser.add_argument("--load", dest="load",
                        help="Loads a trained model, given the model's path.")
    args = parser.parse_args()

    if args.train:
        train_mode()

    if args.load:
        load_mode(args.load)
