from distutils.util import change_root
from typing import Tuple
import numpy as np
from mido import MidiFile
import scipy.io.wavfile as wave


NOTE_OFFSET = 51


def load_midi_file(file: str, sample_rate: int = 44100) -> Tuple:
    """
    This function loads a MIDI file into an easier to use python array.

    :param file: The path pointing at the MIDI file.
    :param sample_rate: The sample rate which has to be represented.
    :returns: A numpy array of the shape [samples note is alive, tone, notes start time] and 
    the duration of the file played in seconds.
    """
    # create the midi file object and the list in which the notes
    # and timestamp were stored
    midi_file = MidiFile(file)
    midi_list = []

    time_register = {}
    note_register = {}

    total_time = 0

    for msg in midi_file:
        # we are only interested in messages containing a note
        if "note" in msg.dict().keys():
            # rise the total playtime
            total_time += msg.time

            if msg.type == "note_on":
                # add the note to the register to measure how long this note has to be played
                time_register[msg.note - NOTE_OFFSET] = total_time
            else:
                # calculate the notes start and delta times
                delta_time = int((total_time - time_register[msg.note - NOTE_OFFSET]) * sample_rate)
                start_time = int(time_register[msg.note - NOTE_OFFSET] * sample_rate)

                # then add the note to the note register to group notes which share the same start time
                note = (delta_time, int(msg.note - NOTE_OFFSET), start_time)
                note_list = note_register.get(start_time, [])
                note_list.append(note)
                note_register[start_time] = note_list

    # calculate the time in seconds by summing all messages times and
    # multiplying it by the sample rate
    return note_register, int(total_time * sample_rate)


def load_wave_file(file: str) -> np.array:
    """
    This function loads a WAVE file into a python array.
    :param file: The path pointing at a WAVE file.
    :returns: Array containing the waveform [left, right] and the 
    duration of the wave form in seconds.
    """
    wave_file = wave.read(file)
    wave_file = np.array(wave_file[1], dtype=float)
    time = len(wave_file) / 44100
    return wave_file, time
