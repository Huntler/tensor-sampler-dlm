import argparse
from data.dataset import MidiWaveDataset
from model.wispy_waterfall import WispyWaterfall
from torch.utils.data import DataLoader
from multiprocessing.spawn import freeze_support
from tqdm import tqdm

def train_mode():
    freeze_support()
    device = "cuda"

    # create the dataset loader
    dataset = MidiWaveDataset(root_dir="dataset/train_0")
    batch_size = int(dataset.sample_rate * 1)# seconds
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    print(f"Training on {int(len(dataset) / batch_size)} batches, each {batch_size} samples " +
          f"({batch_size / dataset.sample_rate} sec) big.")
    
    # create the DLM to use
    # define a quarter of a second as rolling window
    model = WispyWaterfall()
    model.use_device(device)

    # train the model
    for notes_active, wave_sample in tqdm(dataloader):
        notes_active, wave_sample = notes_active.to(device), wave_sample.to(device)
        model.learn(notes_active, wave_sample, epochs=1)
        model.save_to_default()
        print("batch learned")

def load_mode(path):
    # load the model given the path
    model = WispyWaterfall(log=False)
    model.load(path)

    # create the test dataset and execute the model on it
    dataset = MidiWaveDataset(root_dir="dataset/train_1")
    batch_size = int(dataset.sample_rate * 2)# seconds
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    for notes_active, _ in tqdm(dataloader):
        wave_sample = model.predict(notes_active)
        print("batch predicted")

# how to use the dataset MidiWave dataset
# FIXME: rolling window not needed if LSTM are used (NN knows previous behaviour)
# FIXME: remove data/iterators.py afterwards
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load a VST Tensor-Sample DLM.")
    parser.add_argument("--train", action='store_true', default=False, 
                        dest="train", help="Enables the training mode.")
    parser.add_argument("--load", dest="load", help="Loads a trained model, given the model's path.")
    args = parser.parse_args()

    if args.train:
        train_mode()
    
    if args.load:
        load_mode(args.load)

