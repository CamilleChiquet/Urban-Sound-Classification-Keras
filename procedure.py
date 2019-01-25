import audio_processing
from models import *
import numpy as np
from const import *
import matplotlib.pyplot as plt
from models.cnn import basic_cnn


# audio_processing.generate_spectrograms(display_spectrums=False, verbose=False, window_size=4, apply_filters=True)

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

# data = normalize_data(data)

batch_size = 16
epochs = 100
num_classes = labels.shape[-1]

spectrograms_rows = data.shape[1]
spectrograms_columns = data.shape[2]
data = data.reshape(data.shape[0], spectrograms_rows, spectrograms_columns, 1)
input_shape = (spectrograms_rows, spectrograms_columns, 1)

model = basic_cnn(num_classes=num_classes, input_shape=input_shape)

model.save(MODELS_DIR + 'saved_model.h5')

history = model.fit(data, labels, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True, verbose=2)

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
