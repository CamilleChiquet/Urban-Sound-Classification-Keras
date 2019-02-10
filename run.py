from keras.callbacks import ModelCheckpoint
from audio_processing import generate_spectrograms
import numpy as np
from audio_augmentation import generate_noise_augmentation
from const import *
import matplotlib.pyplot as plt
from models.cnn import basic_cnn
import os
from sklearn.preprocessing import LabelBinarizer


FIRST_USE = False


def generate_data_first_use():
    # This filtering method hasn't been effective during my tests
    # audio_processing.generate_filtered_audio(name_prefix="filtered_original-", audio_directory=AUDIO_DIR, output_dir=FILTERED_AUDIO_DIR, window_size=4)
    # audio_processing.generate_spectrograms(name_prefix="filtered_original-", audio_directory=FILTERED_AUDIO_DIR, display_spectrums=False, verbose=False, window_size=4)

    nb_folders = 10
    for i in range(1, nb_folders):
        folder_name = 'fold' + str(i)
        generate_spectrograms(name_prefix="original-" + folder_name + '-',
                              audio_directory=AUDIO_DIR + folder_name + '/', display_spectrums=False,
                              verbose=False, window_size=4)
        generate_noise_augmentation(input_directory=AUDIO_DIR + folder_name + '/',
                                    target_directory=AUGMENTED_AUDIO_DIR + 'noise/' + folder_name)
        generate_spectrograms(name_prefix="noise-" + folder_name + '-',
                              audio_directory=AUGMENTED_AUDIO_DIR + "noise/" + folder_name + '/', display_spectrums=False,
                              verbose=False, window_size=4)


# Shuffles the two arrays the same way, elements at the same indice will always stay in correspondence (import for data and labels arrays)
def shuffle_in_union(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def load_data():
    # We load the first original sounds spectrograms dataset as validation data
    val_data = np.load(SPECTROGRAMS_DIR + 'original-fold10-data.npy')
    val_labels = np.load(SPECTROGRAMS_DIR + 'original-fold10-labels.npy')

    data, labels = None, None

    # The others will be loaded for training
    nb_folders = 10
    for i in range(1, nb_folders):
        folder_name = 'fold' + str(i)
        if data is None:
            data = np.load(SPECTROGRAMS_DIR + 'original-' + folder_name + '-data.npy')
            labels = np.load(SPECTROGRAMS_DIR + 'original-' + folder_name + '-labels.npy')

            data = np.concatenate((data, np.load(SPECTROGRAMS_DIR + 'noise-' + folder_name + '-data.npy')))
            labels = np.concatenate((labels, np.load(SPECTROGRAMS_DIR + 'noise-' + folder_name + '-labels.npy')))
        else:
            data = np.concatenate((data, np.load(SPECTROGRAMS_DIR + 'original-' + folder_name + '-data.npy')))
            labels = np.concatenate((labels, np.load(SPECTROGRAMS_DIR + 'original-' + folder_name + '-labels.npy')))

            data = np.concatenate((data, np.load(SPECTROGRAMS_DIR + 'noise-' + folder_name + '-data.npy')))
            labels = np.concatenate((labels, np.load(SPECTROGRAMS_DIR + 'noise-' + folder_name + '-labels.npy')))
    return data, labels, val_data, val_labels


if FIRST_USE:
    generate_data_first_use()

data, labels, validation_data, validation_labels = load_data()

print(f'data shape : {data.shape}')
print(f'labels shape : {labels.shape}')
print(f'validation data shape : {validation_data.shape}')
print(f'validation labels shape : {validation_labels.shape}')

batch_size = 16
epochs = 200

spectrograms_rows = data.shape[1]
spectrograms_columns = data.shape[2]

# data = normalize_data(data)
data = data.reshape(data.shape[0], spectrograms_rows, spectrograms_columns, 1)
# validation_data = normalize_data(validation_data)
validation_data = validation_data.reshape(validation_data.shape[0], spectrograms_rows, spectrograms_columns, 1)

input_shape = (spectrograms_rows, spectrograms_columns, 1)

# filepath = MODELS_DIR + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
saved_model_path = MODELS_DIR + "best-model.hdf5"
checkpoint = ModelCheckpoint(saved_model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model = basic_cnn(num_classes=NB_CLASSES, input_shape=input_shape)
# model.load_weights(MODELS_DIR + 'best-model.hdf5')
history = model.fit(data, labels, batch_size=batch_size, epochs=epochs,
                    validation_data=(validation_data, validation_labels), shuffle=True, verbose=2,
                    callbacks=callbacks_list)
# model.save(MODELS_DIR + 'saved_model.h5')

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
