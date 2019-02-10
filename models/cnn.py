from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, regularizers, LeakyReLU, ReLU
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import GaussianDropout, GaussianNoise
from keras.utils.generic_utils import get_custom_objects
import keras


# Google's special function
def swish(x):
    return K.sigmoid(x) * x


def basic_cnn(num_classes, input_shape):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(4, 4), activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

# On Urban Sound dataset

# ========== parameters ===========
# spectrograms of shape (92, 193) of 4s audio files
# batch size = 16
# no pre-processing

# ========== without data augmentation ===========
# no data-augmentation, only basic spectrograms of shape (92, 193) of 4s audio files
# 89th iteration accuracy : 0.77460

# ========== with data augmentation ===========
# data-augmentation : 1 file created for each original file with noise added
# 1/10 spectrograms (873) as validation datas (spectrograms from original files, not augmented ones)
# training data : 8732 - 873  original-spectrograms + 8732 - 873 noise-spectrograms
# 182th iteration accuracy : 0.79092
def cnn2(num_classes, input_shape):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(4, 4), activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def test(input_shape, num_labels):
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(4, 4), kernel_initializer='normal', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=32, kernel_size=(4, 4), kernel_initializer='normal', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='normal', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(units=32))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(units=16))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(units=num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(),
              metrics=['accuracy'])

    return model
