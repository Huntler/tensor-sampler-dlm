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
                midi_track[total_time] = one_hot(note_index, note_value, dimension, precision)
                continue

            note_value = -1 if msg.type == "note_off" else 1
            active_notes += one_hot(note_index, note_value, dimension, precision)
            midi_track[total_time] = active_notes
    
    return midi_track


def create_dataframe(src_dir: str, dimension: int, offset: int,
                     sequence_length: int, precision: np.dtype) -> None:
    path = lambda i: f"{src_dir}/dataset_{i}.pandas"

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
    temp_dict = {"Xnotes": [], "Xsamples":[], "y": []}
    X_note, X_sample = [], []
    i = 0
    for time_index, start_time in tqdm(enumerate(start_times), total=len(start_times)):
        # get the correct end time 
        end_time = len(wave)
        if len(start_times) != time_index + 1:
            end_time = start_times[time_index + 1]

        # iterte until the end time
        for sample_index in range(start_time, end_time):
            # if a sequence is completed, add it to the dictionary
            if len(X_note) == sequence_length:
                temp_dict["Xnotes"].append(X_note)
                temp_dict["Xsamples"].append(X_sample)
                temp_dict["y"].append(wave[sample_index])
                X_note.pop(0)
                X_sample.pop(0)
            
            # fill the sequence
            X_note.append(track[start_time])
            X_sample.append(wave[sample_index])

            # write the dictionary to disk every minute, otherwise we got RAM problems
            if len(temp_dict["Xnotes"]) > sample_rate * 60:
                df = pd.DataFrame(temp_dict)
                df.to_hdf(path(i), "midi_wave", mode="a", complib="blosc", complevel=9)
                temp_dict = {"Xnotes": [], "Xsamples":[], "y": []}
                X_note, X_sample = [], []
                i += 1

    df = pd.DataFrame(temp_dict)
    df.to_hdf(path(i), "midi_wave", mode="a", complib="blosc", complevel=9)

if __name__ == "__main__":
    root_dir = "./dataset/train_0"
    dataset_files = []
    for df_file in os.listdir(root_dir):
        if df_file[-6:] == "pandas":
            dataset_files.append(df_file)

    if len(dataset_files) == 0:
        create_dataframe(root_dir, 21, 50, 128, np.float16)

    df = pd.read_hdf(f"{root_dir}/dataset_0.pandas", "midi_wave")
    print(df.tail())
