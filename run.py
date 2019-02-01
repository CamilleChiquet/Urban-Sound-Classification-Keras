from keras.callbacks import ModelCheckpoint
import audio_processing
import numpy as np
from const import *
import matplotlib.pyplot as plt
from models.cnn import basic_cnn


# Decomment if first run, else comment it
# audio_processing.generate_spectrograms(display_spectrums=False, verbose=False, window_size=4, apply_filters=True)

# We load the generated spectrograms as input data
data = np.load(SPECTROGRAMS_DIR + 'data.npy')
labels = np.load(SPECTROGRAMS_DIR + 'labels.npy')

def normalize_data(data):
    min_data_value = np.min(data)
    max_data_value = np.max(data)
    # print(min_data_value)
    # print(max_data_value)
    data -= min_data_value
    data /= (max_data_value - min_data_value)
    return data


batch_size = 16
epochs = 200
num_classes = labels.shape[-1]

spectrograms_rows = data.shape[1]
spectrograms_columns = data.shape[2]

# data = normalize_data(data)
data = data.reshape(data.shape[0], spectrograms_rows, spectrograms_columns, 1)

input_shape = (spectrograms_rows, spectrograms_columns, 1)

# filepath = MODELS_DIR + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath = MODELS_DIR + "best-model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model = basic_cnn(num_classes=num_classes, input_shape=input_shape)
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
