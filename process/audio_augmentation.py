import glob
import os
import random

import librosa
import numpy as np
import tempfile
import math

from librosa.effects import time_stretch, pitch_shift


def tmp_path(ext=''):
    tf = tempfile.NamedTemporaryFile()
    return tf.name + ext


def add_noise(x, power=0.005):
    """
    Add noise to audio data
    """

    return x + np.random.normal(0, power, np.shape(x)[0])


def convolve(x, z, level=0.5):
    """
    Apply convolution to infile using impulse response given
    """

    if(level > 1):
        level = 1
    elif(level < 0):
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


def generate_audio_augmentation():
    parent_dir = '..\\data\\'
    audio_dir = parent_dir + 'audio\\'
    augmented_audio_dir = parent_dir + 'augmented_audio\\'
    sub_dirs = ["fold1", "fold2", "fold3", "fold4", "fold5", "fold6", "fold7", "fold8", "fold9", "fold10"]

    classroom, sr_classroom = librosa.load('sounds\\ir_classroom.wav')
    smartphone, sr_smartphone = librosa.load('sounds\\ir_smartphone_mic.wav')

    for l, sub_dir in enumerate(sub_dirs):
        print(sub_dir)
        if not os.path.exists(os.path.join(augmented_audio_dir, sub_dir)):
            os.makedirs(os.path.join(augmented_audio_dir, sub_dir))
        if not os.path.exists(os.path.join(augmented_audio_dir, sub_dir, 'noise')):
            os.makedirs(os.path.join(augmented_audio_dir, sub_dir, 'noise'))
        if not os.path.exists(os.path.join(augmented_audio_dir, sub_dir, 'conv')):
            os.makedirs(os.path.join(augmented_audio_dir, sub_dir, 'conv'))
        if not os.path.exists(os.path.join(augmented_audio_dir, sub_dir, 'conv')):
            os.makedirs(os.path.join(augmented_audio_dir, sub_dir, 'conv'))
        if not os.path.exists(os.path.join(augmented_audio_dir, sub_dir, 'gain')):
            os.makedirs(os.path.join(augmented_audio_dir, sub_dir, 'gain'))
        if not os.path.exists(os.path.join(augmented_audio_dir, sub_dir, 'time')):
            os.makedirs(os.path.join(augmented_audio_dir, sub_dir, 'time'))
        if not os.path.exists(os.path.join(augmented_audio_dir, sub_dir, 'time')):
            os.makedirs(os.path.join(augmented_audio_dir, sub_dir, 'time'))
        if not os.path.exists(os.path.join(augmented_audio_dir, sub_dir, 'pitch')):
            os.makedirs(os.path.join(augmented_audio_dir, sub_dir, 'pitch'))
        if not os.path.exists(os.path.join(augmented_audio_dir, sub_dir, 'pitch')):
            os.makedirs(os.path.join(augmented_audio_dir, sub_dir, 'pitch'))
        for fn in glob.glob(os.path.join(audio_dir, sub_dir, '*.wav')):
            x, sr1 = librosa.load(fn)

            file_name = fn.split(sep='\\')[-1]

            librosa.output.write_wav(os.path.join(augmented_audio_dir, sub_dir, 'noise') + '\\noise_' + file_name, add_noise(x), sr1)

            librosa.output.write_wav(os.path.join(augmented_audio_dir, sub_dir, 'conv') + '\\classroom_' + file_name, convolve(x, classroom, 0.5), sr1)

            librosa.output.write_wav(os.path.join(augmented_audio_dir, sub_dir, 'conv') + '\\smartphone_' + file_name, convolve(x,smartphone, 0.5), sr1)

            librosa.output.write_wav(os.path.join(augmented_audio_dir, sub_dir, 'gain') + '\\gain10_' + file_name, apply_gain(x, 10), sr1)

            librosa.output.write_wav(os.path.join(augmented_audio_dir, sub_dir, 'time') + '\\time_faster_' + file_name, time_stretch(x, 1.2), sr1)

            librosa.output.write_wav(os.path.join(augmented_audio_dir, sub_dir, 'time') + '\\time_slower_' + file_name, time_stretch(x, 0.88), sr1)

            librosa.output.write_wav(os.path.join(augmented_audio_dir, sub_dir, 'pitch') + '\\pitch_shifting_m2_' + file_name, pitch_shift(x, sr1,-2), sr1)

            librosa.output.write_wav(os.path.join(augmented_audio_dir, sub_dir, 'pitch') + '\\pitch_shifting_p2_' + file_name, pitch_shift(x, sr1,2), sr1)

# generate_audio_augmentation()
