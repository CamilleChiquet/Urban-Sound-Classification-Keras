import glob
import os
from const import *
import librosa
import numpy as np
import tempfile

from librosa.effects import time_stretch, pitch_shift


def tmp_path(ext=''):
    tf = tempfile.NamedTemporaryFile()
    return tf.name + ext


def add_noise(x, power):
    """
	Add noise to audio data
	"""

    return x + np.random.normal(0, power, np.shape(x)[0])


def convolve(x, z, level=0.5):
    """
	Apply convolution to infile using impulse response given
	"""

    if level > 1:
        level = 1
    elif level < 0:
        level = 0
    y = np.convolve(x, z, 'full')[0:x.shape[0]] * level + x * (1 - level)
    return y


def apply_gain(x, gain):
    """
	Apply gain to infile
	"""

    x = np.copy(x)
    x = x * (10 ** (gain / 20.0))
    x = np.minimum(np.maximum(-1.0, x), 1.0)
    return x


def generate_noise_augmentation(target_directory, input_directory):
    audio_file_extension = ".wav"

    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(audio_file_extension):
                audio_path = os.path.join(root, file)
                signal, sample_rate = librosa.load(audio_path, sr=None)

                energy = np.sum(librosa.feature.rmse(y=signal, hop_length=int(sample_rate / 48)))
                # print(f'file : {file} energy : {energy}')

                librosa.output.write_wav(target_directory + '/' + file, add_noise(signal, power=0.001 * energy), sample_rate)


def generate_conv_augmentation(target_directory, input_directory):
    classroom, sr_classroom = librosa.load('ir_classroom.wav')
    smartphone, sr_smartphone = librosa.load('ir_smartphone_mic.wav')

    audio_file_extension = ".wav"

    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(audio_file_extension):
                audio_path = os.path.join(root, file)
                signal, sample_rate = librosa.load(audio_path, sr=None)

                librosa.output.write_wav(target_directory + '/' + file, convolve(signal, classroom, 0.5), sample_rate)
                librosa.output.write_wav(target_directory + '/' + file, convolve(signal, smartphone, 0.5), sample_rate)


def generate_gain_augmentation(target_directory, input_directory):
    audio_file_extension = ".wav"

    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(audio_file_extension):
                audio_path = os.path.join(root, file)
                signal, sample_rate = librosa.load(audio_path, sr=None)

                librosa.output.write_wav(target_directory, apply_gain(signal, 10), sample_rate)


def generate_time_stretch_augmentation(target_directory, input_directory):
    audio_file_extension = ".wav"

    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(audio_file_extension):
                audio_path = os.path.join(root, file)
                signal, sample_rate = librosa.load(audio_path, sr=None)

                librosa.output.write_wav(target_directory, time_stretch(signal, 1.2), sample_rate)

                librosa.output.write_wav(target_directory, time_stretch(signal, 0.88), sample_rate)


def generate_pitch_shifting_augmentation(target_directory, input_directory):
    audio_file_extension = ".wav"

    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(audio_file_extension):
                audio_path = os.path.join(root, file)
                signal, sample_rate = librosa.load(audio_path, sr=None)

                librosa.output.write_wav(target_directory, pitch_shift(signal, sample_rate, -2), sample_rate)

                librosa.output.write_wav(target_directory, pitch_shift(signal, sample_rate, 2), sample_rate)
