import numpy as np
from mido import MidiFile


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
            midi_list.append([value, msg.note, msg.time])

    # convert the created list to a numpy array and normalise all notes
    midi_list = np.asarray(midi_list, dtype=float)
    midi_list[:, 1] = midi_list[:, 1] / np.max(midi_list[:, 1])

    # calculate the time in seconds by summing all messages times and
    # multiplying it by the sample rate
    time = np.sum(midi_list[:, 2], axis=0) * sample_rate
    return midi_list, time
