import tqdm
import math
from typing import List
import numpy as np

from .loading import load_midi_file


MAX_N_NOTES = 20


def midi_iterator(path: str, sample_rate: int = 44100, progress_bar: bool = True) -> np.array:
    """
    This function iterates over the messages provided by a list of midi 
    messages. This method returns the notes played corresponding to a 
    sample. When using enumerate, the index represents the current sample 
    position.

    :param path: Path pointing to the midi file.
    :param sample_rate: Rate of messages provided per second of audio.
    :param progress_bar: Should a progress bar be shown on screen?
    :retruns: The active notes which have to be played. Represented in 
    an array, where the index shows the note and 0/1 off/on.
    """
    note_register, time = load_midi_file(path)
    def iterator():
        active_playing = np.zeros((MAX_N_NOTES,), dtype=np.int)
        
        for sample_index in range(0, time):
            # get notes, if their start time matches the current time
            notes = note_register.get(sample_index, [])

            # then iterate of the notes and add those to the active playing array
            for delta_time, note_index, start_time in notes:
                active_playing[note_index] = delta_time
            
            yield active_playing
            
            # only lower the times of notes that are played
            # not the fastest but effective
            active_playing[active_playing > 0] -= 1
    
    # show a progress bar using tqdm if wanted
    if progress_bar:
        return tqdm.tqdm(iterator(), total=time)
    else:
        return iterator()


def midi_batch_iterator(path: str, batch_size: int = 256, sample_rate: int = 44100,
                        progress_bar: bool = True) -> List[np.array]:
    """
    This function uses midi_iterator() to create batches of samples.
    :param path: Path pointing to the midi file.
    :param batch_size: The size of a batch.
    :param sample_rate: Rate of messages provided per second of audio.
    :param progress_bar: Should a progress bar be shown on screen?
    """
    # load the time, to calculate the amount of batches to iterate over
    _, time = load_midi_file(path)
    n_batches = math.ceil(time / batch_size)

    # define the iterator which creates batches of the given size
    def iterator():
        batch = []
        for sample in midi_iterator(path, sample_rate):
            # create a batch until its size is sufficient
            batch.append(sample)
            if len(batch) == batch_size:
                yield batch
                batch = []

        # in the case there is a batch smaller than the given size, return tit
        yield batch

    # show a progress bar using tqdm if wanted
    if progress_bar:
        return tqdm.tqdm(iterator(), total=n_batches)
    else:
        return iterator()
