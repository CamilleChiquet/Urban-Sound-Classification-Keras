from keras.callbacks import ModelCheckpoint
import audio_processing
import numpy as np
from const import *
import matplotlib.pyplot as plt
from models.cnn import basic_cnn
import os
from sklearn.preprocessing import LabelBinarizer


def generate_data_first_use():
    # audio_processing.generate_filtered_audio(name_prefix="filtered_original-", audio_directory=AUDIO_DIR, output_dir=FILTERED_AUDIO_DIR, window_size=4)
    # audio_processing.generate_spectrograms(name_prefix="filtered_original-", audio_directory=FILTERED_AUDIO_DIR, display_spectrums=False, verbose=False, window_size=4)
    audio_processing.generate_spectrograms(name_prefix="original-", audio_directory=AUDIO_DIR, display_spectrums=False, verbose=False, window_size=4)


FIRST_USE = False
if FIRST_USE:
    generate_data_first_use()

# We load the generated spectrograms as input data
data = np.load(SPECTROGRAMS_DIR + 'original-data.npy')
labels = np.load(SPECTROGRAMS_DIR + 'original-labels.npy')
print(data.shape)
print(labels.shape)

batch_size = 16
epochs = 200

spectrograms_rows = data.shape[1]
spectrograms_columns = data.shape[2]

# data = normalize_data(data)
data = data.reshape(data.shape[0], spectrograms_rows, spectrograms_columns, 1)

input_shape = (spectrograms_rows, spectrograms_columns, 1)

# filepath = MODELS_DIR + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
saved_model_path = MODELS_DIR + "best-model.hdf5"
checkpoint = ModelCheckpoint(saved_model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model = basic_cnn(num_classes=NB_CLASSES, input_shape=input_shape)
# model.load_weights(MODELS_DIR + 'best-model.hdf5')
history = model.fit(data, labels, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True, verbose=2, callbacks=callbacks_list)
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
