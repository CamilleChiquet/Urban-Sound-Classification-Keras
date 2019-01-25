import os
import librosa
import numpy as np
import pickle
import math
from const import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import librosa.display


def convert_category_to_name(category):
	if category == '0':
		return 'air_conditioner'
	elif category == '1':
		return 'car_horn'
	elif category == '2':
		return 'children_playing'
	elif category == '3':
		return 'dog_bark'
	elif category == '4':
		return 'drilling'
	elif category == '5':
		return 'engine_idling'
	elif category == '6':
		return 'gun_shot'
	elif category == '7':
		return 'jackhammer'
	elif category == '8':
		return 'siren'
	elif category == '9':
		return 'street_music'


def visualize_melspectrogram(spectrogram, category):
	librosa.display.specshow(spectrogram, x_axis='time', y_axis='linear')
	plt.colorbar(format='%+2.0f dB')
	plt.title(category)
	plt.show()


def calculate_spectrogram(signal, sample_rate, window_size=4.0, nb_filters=92):

	# We want a fixed window size to give to the cnn
	wanted_length = (window_size * sample_rate)
	signal_length = len(signal)
	# print(f'wanted lenght : {wanted_length} vs signal length : {signal_length}')
	if signal_length > wanted_length:
		signal = signal[:wanted_length]
	elif signal_length < wanted_length:
		signal = np.pad(signal, (0, int(wanted_length - signal_length)), 'constant')
	# print(f'new length : {len(signal)}')

	# Generates the spectrogram
	spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=nb_filters,
												 hop_length=int(sample_rate / 48))
	# spectrogram = librosa.stft(signal, n_fft=512, hop_length=int(sample_rate/150))**2
	spectrogram = librosa.amplitude_to_db(spectrogram)
	return spectrogram

def generate_spectrograms(display_spectrums=False, verbose=True, window_size=4.0, nb_filters=92):
	audio_file_extension = ".wav"
	data = []
	labels = []
	min_spectrogram_length = math.inf
	for root, dirs, files in os.walk(AUDIO_DIR):
		for file in files:
			if file.endswith(audio_file_extension):
				category = file.split("-")[-3]
				category = convert_category_to_name(category)

				file_name = file[:-len(audio_file_extension)]
				audio_path = os.path.join(root, file)

				signal, sample_rate = librosa.load(audio_path, sr=None)

				spectrogram = calculate_spectrogram(signal=signal, sample_rate=sample_rate, window_size=window_size, nb_filters=nb_filters)

				if verbose:
					print(f'{category}, file : {file_name}')
					print(f'sample rate : {sample_rate}')
					print(f'spectrum shape : {spectrogram.shape}')

				if display_spectrums:
					visualize_melspectrogram(spectrogram, category)

				data.append(spectrogram)
				if spectrogram.shape[1] < min_spectrogram_length:
					min_spectrogram_length = spectrogram.shape[1]
				labels.append(category)

	# Despite the window mecanism, there are still small differences between spectrograms 2nd dimension's size
	for i in range(len(data)):
		data[i] = data[i][:, :min_spectrogram_length]

	encoder = LabelBinarizer()
	labels = encoder.fit_transform(labels)
	labels = np.array(labels, dtype=np.int)
	np.save(SPECTROGRAMS_DIR + 'labels.npy', labels)
	# print(f'labels shape : {labels.shape}')

	data = np.array(data)
	np.save(SPECTROGRAMS_DIR + 'data.npy', data)


def save_object(obj, filename):
	with open(filename, 'wb') as output:  # Overwrites any existing file.
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
	with open(filename, 'rb') as input:
		return pickle.load(input)
