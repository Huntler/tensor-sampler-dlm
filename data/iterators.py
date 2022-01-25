from asyncio import FastChildWatcher
from typing import List
import numpy as np


MAX_N_NOTES = 20


def midi_iterator(midi_list: np.array, sample_rate: int = 44100) -> List:
    active_playing = np.zeros((MAX_N_NOTES,), dtype=bool)
    change_sample = 0
    current_time = 0

    # iterate over all midi messages left
    i = 0
    while i != len(midi_list):
        # process the midi information
        sample_length = int(midi_list[i][2] * sample_rate)
        value = int(midi_list[i][0])
        note = int(midi_list[i][1])

        # if the time has not changed, then apply the midi messages
        # to the active_playing notes
        if change_sample == sample_length:
            change_sample = 0
            active_playing[note] = value
            i += 1
            continue

        # there is a time change, so return all activate notes
        for j in range(0, sample_length):
            yield active_playing

        # update the change_sample and reduce the index
        change_sample = int(midi_list[i][2] * sample_rate)
        current_time += midi_list[i][2]