import numpy as np
from PIL import Image
from const import *
from audio_augmentation import *
from audio_processing import *


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.fromstring("RGBA", (w, h), buf.tostring())


def normalize_data(data):
    min_data_value = np.min(data)
    max_data_value = np.max(data)
    # print(min_data_value)
    # print(max_data_value)
    data -= min_data_value
    data /= (max_data_value - min_data_value)
    return data


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

        noise_prefix = 'noise'
        generate_noise_augmentation(input_directory=AUDIO_DIR + folder_name + '/',
                                    target_directory=AUGMENTED_AUDIO_DIR + noise_prefix + '/' + folder_name)
        generate_spectrograms(name_prefix=noise_prefix + "-" + folder_name + '-',
                              audio_directory=AUGMENTED_AUDIO_DIR + noise_prefix + "/" + folder_name + '/', display_spectrums=False,
                              verbose=False, window_size=4)

        #
        # conv_prefix = 'conv'
        # generate_conv_augmentation(input_directory=AUDIO_DIR + folder_name + '/',
        #                            target_directory=AUGMENTED_AUDIO_DIR + conv_prefix + '/' + folder_name)
        # generate_spectrograms(name_prefix=conv_prefix + "-" + folder_name + '-',
        #                       audio_directory=AUGMENTED_AUDIO_DIR + conv_prefix + "/" + folder_name + '/',
        #                       display_spectrums=False,
        #                       verbose=False, window_size=4)
        #
        # conv_prefix = 'pitch'
        # generate_pitch_shifting_augmentation(input_directory=AUDIO_DIR + folder_name + '/',
        #                                      target_directory=AUGMENTED_AUDIO_DIR + conv_prefix + '/' + folder_name)
        # generate_spectrograms(name_prefix=conv_prefix + "-" + folder_name + '-',
        #                       audio_directory=AUGMENTED_AUDIO_DIR + conv_prefix + "/" + folder_name + '/',
        #                       display_spectrums=False,
        #                       verbose=False, window_size=4)
        #
        # conv_prefix = 'time_stretch'
        # generate_time_stretch_augmentation(input_directory=AUDIO_DIR + folder_name + '/',
        #                             target_directory=AUGMENTED_AUDIO_DIR + conv_prefix + '/' + folder_name)
        # generate_spectrograms(name_prefix=conv_prefix + "-" + folder_name + '-',
        #                       audio_directory=AUGMENTED_AUDIO_DIR + conv_prefix + "/" + folder_name + '/', display_spectrums=False,
        #                           verbose=False, window_size=4)


# Shuffles the two arrays the same way, elements at the same indice will always stay in correspondence (import for data and labels arrays)
def shuffle_in_union(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def load_data(augmentations):
    # We load the first original sounds spectrograms dataset as validation data
    val_data = np.load(SPECTROGRAMS_DIR + 'original-fold10-data.npy')
    val_labels = np.load(SPECTROGRAMS_DIR + 'original-fold10-labels.npy')

    data, labels = None, None

    # The others will be loaded for training
    nb_folders = 10
    for i in range(1, nb_folders):
        folder_name = 'fold' + str(i)

        # ========== Orignial data ==============

        if data is None:
            data = np.load(SPECTROGRAMS_DIR + 'original-' + folder_name + '-data.npy')
            labels = np.load(SPECTROGRAMS_DIR + 'original-' + folder_name + '-labels.npy')

        else:
            data = np.concatenate((data, np.load(SPECTROGRAMS_DIR + 'original-' + folder_name + '-data.npy')))
            labels = np.concatenate((labels, np.load(SPECTROGRAMS_DIR + 'original-' + folder_name + '-labels.npy')))

        # =========== Augmented data =============

        for augmentation in augmentations:
            data = np.concatenate((data, np.load(SPECTROGRAMS_DIR + augmentation + '-' + folder_name + '-data.npy')))
            labels = np.concatenate((labels, np.load(SPECTROGRAMS_DIR + augmentation + '-' + folder_name + '-labels.npy')))
    return data, labels, val_data, val_labels
