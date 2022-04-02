from data.dataset import MidiWaveDataset
from torch.utils.data import DataLoader
from multiprocessing.spawn import freeze_support

# how to use the dataset MidiWave dataset
# FIXME: put this into the training loop
# FIXME: remove data/iterators.py afterwards
if __name__ == '__main__':
    freeze_support()

    dataset = MidiWaveDataset(root_dir="dataset/train_0")
    dataloader = DataLoader(dataset, batch_size=44100, shuffle=False, num_workers=8)

    for i, sample_batched in enumerate(dataloader):
        notes_active, wave_sample = sample_batched
        print(i)