import numpy as np
from mido import MidiFile
import scipy.io.wavfile as wave


NOTE_OFFSET = 51


def load_midi_file(file: str, sample_rate: int = 44100) -> np.array:
    """
    This function loads a MIDI file into an easier to use python array.

    :param file: The path pointing at the MIDI file.
    :param sample_rate: The sample rate which has to be represented.
    :returns: A numpy array of the shape [note on/off, tone, time] and 
    the duration of the file played in seconds.
    """
    # create the midi file object and the list in which the notes
    # and timestamp were stored
    midi_file = MidiFile(file)
    midi_list = []

    for msg in midi_file:
        # we are only interested in messages containing a note
        if "note" in msg.dict().keys():
            value = 1 if msg.type == "note_on" else 0
            midi_list.append([value, msg.note - NOTE_OFFSET, msg.time])

    # convert the created list to a numpy array
    midi_list = np.asarray(midi_list, dtype=float)

    # calculate the time in seconds by summing all messages times and
    # multiplying it by the sample rate
    time = np.sum(midi_list[:, 2], axis=0) * sample_rate
    return midi_list, int(time)


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
