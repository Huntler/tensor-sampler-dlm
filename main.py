import argparse
import numpy as np
import scipy.io.wavfile
import torch
import os
from copy import deepcopy
from data.audioset import AudioDataset
from model.base import BaseModel
from utils.config import config
from torch.utils.data import DataLoader
from multiprocessing.spawn import freeze_support
from tqdm import tqdm

config_dict = None


def prepare_dataset() -> DataLoader:
    # get the related configuration
    dataset_dict = deepcopy(config_dict)["dataset"]
    dataset_dict["prev_samples"] = config_dict["prev_samples"]
    dataset_dict["future_samples"] = config_dict["future_samples"]
    dataset_dict["precision"] = np.float16 if config_dict["device"] == "cuda" else np.float32
    loader_dict = dataset_dict["loader"]
    del dataset_dict["loader"]

    # create the dataset and dataset loader
    dataset = AudioDataset(**dataset_dict)
    dataloader = DataLoader(dataset, **loader_dict)

    # give some info
    name = dataset_dict["name"]
    print(f"Loaded dataset {name} with size {len(dataset)}")

    return dataloader


def prepare_model(sequence_length, channels) -> BaseModel:
    # get the related configuration
    model_dict = deepcopy(config_dict)
    model_name = model_dict["model"]["name"]
    model_dict["model"]["log"] = model_dict["log"]
    model_dict["precision"] = np.float16 if config_dict["device"] == "cuda" else np.float32
    model_dict["model"]["sequence_length"] = sequence_length
    model_dict["model"]["channels"] = channels
    del model_dict["model"]["name"]
    del model_dict["model"]["train"]

    # get the model class and create an instance of it
    model_class = config.get_model(model_name)
    model: BaseModel = model_class(**model_dict["model"])
    model.use_device(config_dict["device"])

    # store the used config if logging is enabled
    model_dict["evaluation"] = model.log_path if config_dict["log"] else None
    if config_dict["log"]:
        to_store = deepcopy(config_dict)
        to_store["log"] = False
        to_store["evaluation"] = model.log_path
        model_dict["evaluation"] = model.log_path
        to_store["device"] = "cpu"
        config.store_args(f"{model.log_path}/config.yml", to_store)
    
    else:
        # load the model given the path
        path = []
        root_folder = config_dict["evaluation"]  
        for file in os.listdir(root_folder):
            if ".torch" in file:
                path.append(file)
        if len(path) == 0:
            print("No model to evaluate.")
        
        path = f"{root_folder}/{max(path)}"
        model.log_path = root_folder
        model.load(path)


    # give some infos
    print(f"Loaded model {model_name}")

    return model


def train_mode():
    train_dict = config_dict["model"]["train"]

    dataloader = prepare_dataset()
    model = prepare_model(dataloader.dataset.sequence_length, dataloader.dataset.channels)

    model.train_on(dataloader, **train_dict)
    model.save_to_default()


def load_mode():
    # disable some configurations, such as shuffle to gain a useful output
    config_dict["log"] = False
    config_dict["dataset"]["loader"]["shuffle"] = False
    
    # prepare the dataset and model
    dataloader = prepare_dataset()
    model = prepare_model(dataloader.dataset.sequence_length, dataloader.dataset.channels)
    root_folder = model.log_path

    # use the loaded model to predict a waveform
    # FIXME: Prediction speed much slower than training, difference to training: model loaded on CPU instead of GPU
    wave = np.zeros((1, 2))
    for X, y in tqdm(dataloader):
        window = model.predict(X)
        window = dataloader.dataset.convert(window)
        wave = np.append(wave, window, axis=0)

    print(wave, wave.shape)
    scipy.io.wavfile.write(f"{root_folder}/output.wav", 44100, wave)


# how to use the dataset MidiWave dataset
if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser(
        description="Train or load a VST Tensor-Sample DLM.")
    parser.add_argument("--config", dest="config",
                        help="Configuration of the model.")
    parser.add_argument("--load-only", dest="load_only", action='store_true',
                        help="Only loads the model and processes a MIDI file.")
    parser.add_argument("--debug", dest="debug", help="Starts the program in a given mode to debug easier. (e.g. [dataset])")
    args = parser.parse_args()

    # load model arguments and check if it can be evaluated
    config_dict = config.get_args(args.config)
    log_path = config_dict["evaluation"]

    if args.debug == "dataset":
        dataloader = prepare_dataset()
        for X, y in tqdm(dataloader):
            pass

        for X, y in tqdm(dataloader.dataset):
            pass

        quit()

    if args.load_only:
        if log_path is None:
            raise RuntimeError("Expected key 'evaluation' in config. Unable to load model.")
        
        load_mode()
        quit()

    if log_path is None:
        train_mode()
        load_mode()

    if log_path:
        load_mode()
