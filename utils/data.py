from typing import Dict
import numpy as np
from mido import MidiFile


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