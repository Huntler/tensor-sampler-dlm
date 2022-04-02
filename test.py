from model.wispy_waterfall import WispyWaterfall
from data.loading import load_midi_file, load_wave_file
from data.iterators import midi_iterator


wave = load_wave_file("dataset/train.wav")[0]
midi = midi_iterator("dataset/train.mid")

model = WispyWaterfall()

model.learn(midi, wave, epochs=5)
model.test(midi, wave)