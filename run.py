from keras.callbacks import ModelCheckpoint
from const import *
import matplotlib.pyplot as plt
from models.cnn import cnn
from utils import generate_data_first_use, load_data
import numpy as np


# If it's your first use, set it to True in order to generate the input data (spectrograms) from the audio files
FIRST_USE = False

if FIRST_USE:
    generate_data_first_use()

# Data augmentations to load
# augmentations = ['noise', 'conv', 'pitch', 'time_stretch']
augmentations = ['noise']
data, labels, validation_data, validation_labels = load_data(augmentations)

print(f'\ndata shape : {data.shape}')
print(f'labels shape : {labels.shape}')
print(f'\nvalidation data shape : {validation_data.shape}')
print(f'validation labels shape : {validation_labels.shape}')

batch_size = 16
epochs = 150

spectrograms_rows = data.shape[1]
spectrograms_columns = data.shape[2]

# Normalization didn't improve my results, that's why it's commented
# data = normalize_data(data)
# validation_data = normalize_data(validation_data)

data = data.reshape(data.shape[0], spectrograms_rows, spectrograms_columns, 1)
validation_data = validation_data.reshape(validation_data.shape[0], spectrograms_rows, spectrograms_columns, 1)

input_shape = (spectrograms_rows, spectrograms_columns, 1)

# We save the best model we will see during the training(s)
saved_model_path = MODELS_DIR + "best-model.hdf5"
checkpoint = ModelCheckpoint(saved_model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Training a neural network is a stochastic process. It means there is randomness that affects the results.
# As a consequence, to have a good idea of our network's global performance on our problem,
# we make multiple trainings (each one with a different set of weights and biases at starting point).
# Set it to 1 if you only want to train your network once.
nb_trainings = 10
trainings_accuracy = []
for i in range(nb_trainings):
    # We load a new model
    model = cnn(num_classes=NB_CLASSES, input_shape=input_shape)
    # Start training
    history = model.fit(data, labels, batch_size=batch_size, epochs=epochs,
                        validation_data=(validation_data, validation_labels), shuffle=True, verbose=2,
                        callbacks=callbacks_list)
    max_acc = max(history.history['val_acc'])
    trainings_accuracy.append(max_acc)
    print(f'============================')
    print(f'Training best acc : {max_acc}')
    print(f'Total accuracies : {trainings_accuracy}')
    print(f'============================')

trainings_accuracy = np.array(trainings_accuracy)
print(f'Training session max acc : {np.max(trainings_accuracy)}')
print(f'Training session min acc : {np.min(trainings_accuracy)}')
print(f'Training session mean acc : {np.mean(trainings_accuracy)}')

plt.boxplot(trainings_accuracy)
plt.title('Training session accuracy')
plt.ylabel('accuracy')
plt.show()
