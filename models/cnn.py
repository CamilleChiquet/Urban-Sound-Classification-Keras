from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, regularizers, LeakyReLU, ReLU
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import GaussianDropout, GaussianNoise
from keras.utils.generic_utils import get_custom_objects
import keras


# Google's activation function
def swish(x):
    return K.sigmoid(x) * x


# On Urban Sound dataset

# ========== parameters ===========
# spectrograms of shape (92, 193) of 4s audio files
# batch size = 16
#
# data-augmentation : 1 file created for each original file with white noise added to them.
# 1/10 spectrograms (873) as validation datas (spectrograms from original files, not augmented ones)
# training data : (8732 - 873)  original-spectrograms + (8732 - 873) noise-spectrograms
#
# On 10 trainings :
#   min acc : 80.3%
#   mean acc : 82.6%
#   max acc : 84.3%
def cnn(num_classes, input_shape):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(40, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))


def cnn_test(num_classes, input_shape):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu', input_shape=input_shape, padding='same', kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu', input_shape=input_shape, padding='same', kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(40, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model
