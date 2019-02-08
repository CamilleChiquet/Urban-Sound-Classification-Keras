import os
import librosa
import numpy as np
import pickle
import math

from sklearn.preprocessing import LabelBinarizer

from const import *
import matplotlib.pyplot as plt
import librosa.display
from sklearn.ensemble import IsolationForest


def convert_category_to_name(category):
	category = int(category)
	if category >= NB_CLASSES:
		raise ValueError(f"The category : {category} doesn't exist")
	return CLASSES[category]


def visualize_melspectrogram(spectrogram, category):
	librosa.display.specshow(spectrogram, x_axis='time', y_axis='linear')
	plt.colorbar(format='%+2.0f dB')
	plt.title(category)
	plt.show()


def calculate_spectrogram(signal, sample_rate, window_size=4.0, nb_filters=92):
	# We want a fixed window size to give to the cnn
	wanted_length = int(window_size * sample_rate)
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


def generate_spectrograms(audio_directory=AUDIO_DIR, display_spectrums=False, verbose=False, window_size=4.0,
						  nb_filters=92, name_prefix='', audio_file_extension=".wav", output_dir=SPECTROGRAMS_DIR):
	'''

	:param audio_directory: directory where it gets the audio files to analyze
	:param display_spectrums:
	:param verbose:
	:param window_size: size in seconds of the wanted spectrograms
	:param nb_filters: number of rows for spectrograms
	:param name_prefix: names' prefix of the generated data
	:param audio_file_extension: type of audio files
	:return:
	'''
	data = []
	labels = []
	min_spectrogram_length = math.inf
	for root, dirs, files in os.walk(audio_directory):
		for file in files:
			if file.endswith(audio_file_extension):
				category = file.split("-")[-3]
				category = convert_category_to_name(category)

				file_name = file[:-len(audio_file_extension)]
				audio_path = os.path.join(root, file)

				signal, sample_rate = librosa.load(audio_path, sr=None)

				spectrogram = calculate_spectrogram(signal=signal, sample_rate=sample_rate, window_size=window_size,
													nb_filters=nb_filters)

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

	## Despite the window mecanism, there are still small differences between spectrograms 2nd dimension's size
	for i in range(len(data)):
		data[i] = data[i][:, :min_spectrogram_length]

	encoder = LabelBinarizer()
	labels = encoder.fit_transform(labels)
	labels = np.array(labels, dtype=np.int)
	np.save(output_dir + name_prefix + 'labels.npy', labels)
	# print(f'labels shape : {labels.shape}')

	data = np.array(data)
	np.save(output_dir + name_prefix + 'data.npy', data)


def save_object(obj, filename):
	with open(filename, 'wb') as output:  # Overwrites any existing file.
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
	with open(filename, 'rb') as input:
		return pickle.load(input)


def generate_filtered_audio(audio_directory=AUDIO_DIR, output_dir=FILTERED_AUDIO_DIR, window_size=4.0, nb_filters=92,
								   audio_file_extension=".wav", name_prefix=''):
	'''
	Generates spectrograms for each audio file in the given audio_directory.
	A selection is done on the spectrums in order to delete those who are too different from the others, inside a same category.
	Ex : some audio files in "dog_bark" category contains only sounds of crickets. They will be detected and not kept by
		the algo.

	:param audio_directory: directory where it gets the audio files to analyze
	:param window_size: size in seconds of the wanted spectrograms
	:param nb_filters: number of rows for spectrograms
	:param audio_file_extension: type of audio files
	:param name_prefix: names' prefix of the generated data
	:return: spectrograms without ousiders
	'''
	for _, category_to_analyze in CLASSES.items():
		data = []
		files_path = []
		min_spectrogram_length = math.inf
		for root, dirs, files in os.walk(audio_directory):
			for file in files:
				if file.endswith(audio_file_extension):
					category = file.split("-")[-3]
					category = convert_category_to_name(category)
					if category == category_to_analyze:
						audio_path = os.path.join(root, file)

						signal, sample_rate = librosa.load(audio_path, sr=None)

						spectrogram = calculate_spectrogram(signal=signal, sample_rate=sample_rate,
															window_size=window_size, nb_filters=nb_filters)

						data.append(spectrogram)
						if spectrogram.shape[1] < min_spectrogram_length:
							min_spectrogram_length = spectrogram.shape[1]
						files_path.append(audio_path)

		for i in range(len(data)):
			data[i] = data[i][:, :min_spectrogram_length]
		data = np.array(data)

		nb_spectrums = data.shape[0]

		clf = None
		if category_to_analyze == 'air_conditioner':
			clf = IsolationForest(behaviour='new', max_samples=nb_spectrums, contamination='auto', n_estimators=100,
								  max_features=1.0)
		elif category_to_analyze == 'car_horn':
			clf = IsolationForest(behaviour='new', max_samples=nb_spectrums, contamination=0.03, n_estimators=100,
								  max_features=1.0)
		elif category_to_analyze == 'children_playing':
			clf = IsolationForest(behaviour='new', max_samples=nb_spectrums, contamination='auto', n_estimators=100,
								  max_features=1.0)
		elif category_to_analyze == 'dog_bark':
			clf = IsolationForest(behaviour='new', max_samples=nb_spectrums, contamination='auto', n_estimators=100,
								  max_features=1.0)
		elif category_to_analyze == 'engine_idling':
			clf = IsolationForest(behaviour='new', max_samples=nb_spectrums, contamination=0.030, n_estimators=100,
								  max_features=0.1)
		elif category_to_analyze == 'siren':
			clf = IsolationForest(behaviour='new', max_samples=nb_spectrums, contamination=0.03, n_estimators=100,
								  max_features=1.0)
		elif category_to_analyze == 'street_music':
			clf = IsolationForest(behaviour='new', max_samples=nb_spectrums, contamination=0.02, n_estimators=100,
								  max_features=0.1)

		if clf is None:
			for file_path in files_path:
				signal, sample_rate = librosa.load(file_path, sr=None)
				file_name = file_path.split('\\')[-1]
				librosa.output.write_wav(output_dir + file_name, signal, sample_rate)
			continue

		isolation_forest_data = data.copy().reshape((data.shape[0], data.shape[1] * data.shape[2]))

		clf.fit(isolation_forest_data)

		y_pred_train = clf.predict(isolation_forest_data)

		for i in range(len(y_pred_train)):
			if y_pred_train[i] == 1:
				signal, sample_rate = librosa.load(files_path[i], sr=None)
				file_name = files_path[i].split('\\')[-1]
				librosa.output.write_wav(output_dir + file_name, signal, sample_rate)

# air conditioner	good		IsolationForest(behaviour='new', max_samples=data.shape[0], contamination='auto', n_estimators=100, max_features=1.0)
# car horn 			good		IsolationForest(behaviour='new', max_samples=data.shape[0], contamination=0.03, n_estimators=100, max_features=1.0)
# children playing 	good		IsolationForest(behaviour='new', max_samples=data.shape[0], contamination='auto', n_estimators=100, max_features=1.0)
# dog barks 		very good	IsolationForest(behaviour='new', max_samples=data.shape[0], contamination='auto', n_estimators=100, max_features=1.0)
# drilling			medium
# engine idling		good		IsolationForest(behaviour='new', max_samples=data.shape[0], contamination=0.030, n_estimators=100, max_features=0.1)
# gun shot			bad
# jackhammer		extremely bad
# siren				medium 		IsolationForest(behaviour='new', max_samples=data.shape[0], contamination=0.03, n_estimators=100, max_features=1.0)
# street music		good		IsolationForest(behaviour='new', max_samples=data.shape[0], contamination=0.02, n_estimators=100, max_features=0.1)
