import argparse
import numpy as np
import scipy.io.wavfile
import torch
import os
from data.dataset import MidiWaveDataset
from model.base import BaseModel
from utils.audio import play_audio, plot_specgram, plot_waveform, print_stats
from utils.config import config
from torch.utils.data import DataLoader
from multiprocessing.spawn import freeze_support
from tqdm import tqdm

config_dict = None
# FIXME: implement batch normalization (Lecture 2, Slide 102-103)
# FIXME: Normalize using transformers
# FIXME: remove data/iterators.py afterwards
# FIXME: test it


def train_mode():
    # define parameters
    device = config_dict["device"]
    precision = torch.float16 if device == "cuda" else torch.float32

    freeze_support()

    # create the dataset loader
    dataset_name = config_dict["dataset"]["name"]
    dataset = MidiWaveDataset(
        root_dir=f"dataset/{dataset_name}", precision=precision)
    batch_size = int(dataset.sample_rate *
                     config_dict["model"]["batch_size_in_seconds"])

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=config_dict["dataset"]["worker"])
    print(f"Training on {int(len(dataset) / batch_size)} batches, each {batch_size} samples " +
          f"({batch_size / dataset.sample_rate} sec) big.")

    # create the model and train it, if epochs > 0
    epochs = config_dict["model"]["epochs"]
    if epochs == 0:
        return

    # create the DLM to use
    model_class = config.get_model(name=config_dict["model"]["name"])
    model: BaseModel = model_class(precision=precision)
    config_dict["evaluation"] = model.log_path
    model.use_device(device)

    # train the model
    for epoch in range(epochs):
        for notes_active, wave_sample in tqdm(dataloader):
            notes_active, wave_sample = notes_active.to(
                device), wave_sample.to(device)
            model.learn(notes_active, wave_sample, epochs=3)

    model.save_to_default()


def load_mode():
    # load the model given the path
    path = []
    root_folder = config_dict["evaluation"]  
    for file in os.listdir(root_folder):
        if ".torch" in file:
            path.append(file)
    if len(path) == 0:
        print("No model to evaluate.")

    
    path = f"{root_folder}/{max(path)}"

    precision = torch.float16 if config_dict["device"] == "cuda" else torch.float32
    model_class = config.get_model(name=config_dict["model"]["name"])
    model: BaseModel = model_class(precision=precision, log=False)
    model.load(path)

    # create the test dataset and execute the model on it
    dataset_name = config_dict["dataset"]["name"]
    dataset = MidiWaveDataset(root_dir=f"dataset/{dataset_name}", precision=precision)
    batch_size = int(dataset.sample_rate * config_dict["model"]["batch_size_in_seconds"])  # seconds
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=config_dict["dataset"]["worker"])

    # predict the waveform
    wave = []
    for notes_active, _ in tqdm(dataloader):
        window = model.predict(notes_active)
        for sample in window:
            wave.append(sample.numpy())

    wave = np.array(wave).T
    print(wave, wave.shape)
    scipy.io.wavfile.write(f"{root_folder}/{dataset_name}.wav", dataset.sample_rate, wave.T)


# how to use the dataset MidiWave dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or load a VST Tensor-Sample DLM.")
    parser.add_argument("--config", dest="config",
                        help="Configuration of the model.")
    args = parser.parse_args()

    # load arguments and start training
    config_dict = config.get_args(args.config)
    train_mode()

    # move config file to log folder and execute the training mode
    log_path = config_dict["evaluation"]
    if log_path:
        config.store_args(f"{log_path}/config.yml", config_dict)
        load_mode()
