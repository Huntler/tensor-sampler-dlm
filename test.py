from model.wispy_waterfall import WispyWaterfall
from data.loading import load_wave_file
from data.iterators import midi_iterator


wave = load_wave_file("dataset/test.wav")[0]
midi = midi_iterator("dataset/test_0.mid")

model = WispyWaterfall()
model.learn(midi, wave, epochs=5)