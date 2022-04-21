import os
from typing import Dict
import numpy as np
import pandas as pd
import torchaudio
from mido import MidiFile
from tqdm import tqdm


def one_hot(index: int, value: int, dimension: int,
            precision: np.dtype) -> np.array:
    one_hot = np.zeros((dimension), dtype=precision)
    one_hot[index] = value
    return one_hot


def create_midi_track(midi_file: MidiFile, sample_rate: int,
                      offset: int, dimension: int, precision: np.dtype) -> Dict:
    midi_track = {}
    total_time = 0
    for msg in midi_file:
        # we are only interested in messages containing a note
        if "note" in msg.dict().keys():
            # rise the total playtime
            total_time += int(msg.time * sample_rate)
            note_index = msg.note - offset
            active_notes = midi_track.get(total_time, [])

            # if there are no active notes registered for the given time stamp
            if active_notes == []:
                note_value = 0 if msg.type == "note_off" else 1
                midi_track[total_time] = [note_index]#one_hot(note_index, note_value, dimension, precision)
                continue

            note_value = -1 if msg.type == "note_off" else 1
            active_notes += [note_index]#one_hot(note_index, note_value, dimension, precision)
            midi_track[total_time] = active_notes
    
    return midi_track


def create_dataframe(src_dir: str, dimension: int, offset: int,
                     sequence_length: int, precision: np.dtype, 
                     append: bool = False) -> None:
    path = lambda i: f"{src_dir}/dataset_{i}.pandas"
    #if os.path.exists(path):
    #    os.remove(path)

    # load the wave file
    metadata = torchaudio.info(f"{src_dir}/output.wav")
    wave, sample_rate = torchaudio.load(src_dir + "/output.wav")
    wave = np.array(wave.T.numpy(), dtype=np.float16)

    # load the midi file
    midi_file = MidiFile(f"{src_dir}/input.mid")
    track = create_midi_track(midi_file, sample_rate,
                              offset, dimension, np.uint8)
    start_times = [_ for _ in track.keys()]
    start_times.sort()

    # create the dataframe
    # TODO: apply sequences already
    temp_dict = {"active_notes": [], "wave": []}
    i = 0
    for time_index, start_time in tqdm(enumerate(start_times), total=len(start_times)):
        end_time = len(wave)
        if len(start_times) != time_index + 1:
            end_time = start_times[time_index + 1]

        for sample_index in range(start_time, end_time):
            sample = track[start_time]

            temp_dict["active_notes"].append(sample)
            temp_dict["wave"].append(wave[sample_index])

            if len(temp_dict["active_notes"]) > sample_rate * 60:
                _df = pd.DataFrame(temp_dict)
                _df.to_hdf(path(i), "midi_wave", mode="a", complib="blosc", complevel=9)
                del _df
                temp_dict = {"active_notes": [], "wave": []}
                i += 1

    _df = pd.DataFrame(temp_dict)
    _df.to_hdf(path(i), "midi_wave", mode="a", complib="blosc", complevel=9)

if __name__ == "__main__":
    root_dir = "./dataset/train_0"
    if not os.path.exists(f"{root_dir}/dataset.pandas"):
        create_dataframe(root_dir, 21, 50, 0, np.float16)

    df = pd.read_hdf(f"{root_dir}/dataset.pandas", "midi_wave")
    print(df.tail())
    input()
    print(len(df))
