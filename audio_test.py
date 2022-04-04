from data.dataset import MidiWaveDataset
from model.wispy_waterfall import WispyWaterfall
from torch.utils.data import DataLoader
from multiprocessing.spawn import freeze_support
from tqdm import tqdm

# how to use the dataset MidiWave dataset
# FIXME: put this into the training loop
# FIXME: remove data/iterators.py afterwards
if __name__ == '__main__':
    freeze_support()
    device = "cuda"

    # create the dataset loader
    dataset = MidiWaveDataset(root_dir="dataset/train_0")

    batch_size = int(dataset.sample_rate * 2)# seconds
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    print(f"Training on {int(len(dataset) / batch_size)} batches, each {batch_size} samples " +
          f"({batch_size / dataset.sample_rate} sec) big.")
    
    # create the DLM to use
    # define a quarter of a second as rolling window
    model = WispyWaterfall(rolling_window_size=11025)
    model.use_device(device)

    # train the model
    for notes_active, wave_sample in tqdm(dataloader):
        notes_active, wave_sample = notes_active.to(device), wave_sample.to(device)
        model.learn(notes_active, wave_sample, epochs=2)
        model.save_to_default()
        print("batch learned")