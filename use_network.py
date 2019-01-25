from tkinter import filedialog
import tkinter as tk
from PIL import ImageTk, Image
import librosa
from keras.engine.saving import load_model
import matplotlib.pyplot as plt
from matplotlib import cm
from audio_processing import calculate_spectrogram, visualize_melspectrogram
import numpy as np
from PIL import ImageOps, Image
from const import MODELS_DIR
import os


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


def playAudioFile():
    os.system("start " + audio_file)


def selectAudioFile():
    global audio_file
    root.filename = filedialog.askopenfilename(
        initialdir="C:\\Users\\NNED\\PycharmProjects\\urban_sound_classification\\data\\audio\\fold1",
        title="Select file", filetypes=(("wav files", "*.wav"), ("all files", "*.*")))
    audio_file = root.filename

    signal, sample_rate = librosa.load(root.filename, sr=None)

    spectrogram = calculate_spectrogram(signal=signal, sample_rate=sample_rate)
    # visualize_melspectrogram(spectrogram, 'test')
    normalized_spectrogram = normalize_data(spectrogram)
    normalized_spectrogram = np.flip(normalized_spectrogram, 0)

    pilImage = Image.fromarray(np.uint8(normalized_spectrogram * 255), 'L')

    colored_Image = ImageOps.colorize(image=pilImage, black='#000000', mid='#cc5015', white='#ffeeaa')

    width, height = colored_Image.size
    maxsize = (width * 3, height * 3)
    resized_image = colored_Image.resize(maxsize, Image.BOX)

    global spectrogram_image
    global prediction_plot
    global spectrogram_panel
    global prediction_panel
    spectrogram_image = ImageTk.PhotoImage(resized_image)
    spectrogram_panel.configure(image=spectrogram_image)
    spectrogram_panel.pack(side="top", fill="both", expand="yes")

    model = load_model(MODELS_DIR + 'saved_model.h5')
    data = np.reshape(spectrogram, (1, spectrogram.shape[-2], spectrogram.shape[-1], 1))
    prediction = model.predict(data, verbose=2)[0]

    classes = ['Air conditioner', 'Car horn', 'Children playing', 'Dog bark', 'Drilling', 'Engine idling', 'Gun shot',
               'Jackhammer', 'Siren', 'Street music']

    colors = cm.plasma(prediction / float(max(prediction)))
    plot = plt.scatter(prediction, prediction, c=prediction, cmap='plasma')
    plt.clf()
    plt.colorbar(plot)
    plt.bar(classes, prediction, color=colors)
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.25)

    plot_path = 'plot.png'
    plt.savefig(plot_path)

    plot_img = Image.open(plot_path)
    prediction_plot = ImageTk.PhotoImage(plot_img)
    prediction_panel.configure(image=prediction_plot)
    prediction_panel.pack(side="top", fill="both", expand="yes")


root = tk.Tk()
spectrogram_image = None
prediction_plot = None

button_filedialog = tk.Button(root, text="Select audio", command=selectAudioFile)
button_filedialog.pack(pady=3)

button_playaudio = tk.Button(root, text="Play audio", command=playAudioFile)
button_playaudio.pack(pady=3)

label_audiofile = tk.Label(root, text="audio file")
label_audiofile.pack(pady=3)

spectrogram_panel = tk.Label(root, image=spectrogram_image)
prediction_panel = tk.Label(root, image=prediction_plot)
audio_file = None

root.mainloop()
