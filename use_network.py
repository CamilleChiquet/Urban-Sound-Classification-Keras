from tkinter import filedialog
import tkinter as tk
from PIL import ImageTk
import librosa
import matplotlib.pyplot as plt
from matplotlib import cm
from audio_processing import calculate_spectrogram, convert_category_to_name
import numpy as np
from PIL import ImageOps, Image
import os
from const import *
from models.cnn import basic_cnn
from utils import normalize_data



class MyGUI:
    def __init__(self):
        self.root = tk.Tk()

        self.menubar = tk.Menu(self.root)
        self.menubar.add_command(label="Quit", command=self.root.quit)
        self.menubar.add_command(label="Select audio", command=self.selectAudioFile)
        self.menubar.add_command(label="Play current audio", command=self.playAudioFile)

        # display the menu
        self.root.config(menu=self.menubar)

        self.spectrogram_image = None
        self.prediction_plot = None
        self.audio_file = None
        self.model = None

        self.label_audiofile = tk.Label(self.root, text="", font="Helvetica 16 bold")
        self.label_audiofile.pack(pady=3)

        self.spectrogram_panel = tk.Label(self.root, image=self.spectrogram_image)
        self.spectrogram_panel.pack(pady=3)

        self.label_prediction = tk.Label(self.root, text="", font="Helvetica 16 bold")
        self.label_prediction.pack(pady=3)

        self.prediction_panel = tk.Label(self.root, image=self.prediction_plot)
        self.prediction_panel.pack(pady=3)




        self.root.mainloop()

    def playAudioFile(self):
        os.system("start " + self.audio_file)

    def selectAudioFile(self):
        self.root.filename = filedialog.askopenfilename(
            initialdir="C:\\Users\\NNED\\PycharmProjects\\urban_sound_classification\\data\\audio\\fold1",
            title="Select file", filetypes=(("wav files", "*.wav"), ("all files", "*.*")))
        self.audio_file = self.root.filename

        category = self.audio_file.split("-")[-3]
        self.label_audiofile['text'] = 'Category : ' + convert_category_to_name(category).replace('_', ' ')

        signal, sample_rate = librosa.load(self.root.filename, sr=None)

        spectrogram = calculate_spectrogram(signal=signal, sample_rate=sample_rate)
        normalized_spectrogram = normalize_data(spectrogram.copy())
        normalized_spectrogram = np.flip(normalized_spectrogram, 0)

        pilImage = Image.fromarray(np.uint8(normalized_spectrogram * 255), 'L')

        colored_Image = ImageOps.colorize(image=pilImage, black='#000000', mid='#cc5015', white='#ffeeaa')

        width, height = colored_Image.size
        maxsize = (width * 3, height * 3)
        resized_image = colored_Image.resize(maxsize, Image.BOX)

        self.spectrogram_image = ImageTk.PhotoImage(resized_image)
        self.spectrogram_panel.configure(image=self.spectrogram_image)
        self.spectrogram_panel.pack(side="top", fill="both", expand="yes")

        data = np.reshape(spectrogram, (1, spectrogram.shape[-2], spectrogram.shape[-1], 1))

        spectrograms_rows = spectrogram.shape[-2]
        spectrograms_columns = spectrogram.shape[-1]
        input_shape = (spectrograms_rows, spectrograms_columns, 1)

        if self.model is None:
            self.model = basic_cnn(NB_CLASSES, input_shape)
            self.model.load_weights(MODELS_DIR + 'best-model.hdf5')

        y_probabilities = self.model.predict(data, verbose=2)[0]
        y_predicted_classe = y_probabilities.argmax(axis=-1)
        prediction = 'Prediction : ' + convert_category_to_name(y_predicted_classe)

        self.label_prediction['text'] = prediction

        classes = ['Air conditioner', 'Car horn', 'Children playing', 'Dog bark', 'Drilling', 'Engine idling',
                   'Gun shot', 'Jackhammer', 'Siren', 'Street music']

        colors = cm.plasma(y_probabilities / float(max(y_probabilities)))
        plot = plt.scatter(y_probabilities, y_probabilities, c=y_probabilities, cmap='plasma')
        plt.clf()
        plt.colorbar(plot)
        plt.bar(classes, y_probabilities, color=colors)
        plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.25)

        plot_path = 'plot.png'
        plt.savefig(plot_path)

        plot_img = Image.open(plot_path)
        self.prediction_plot = ImageTk.PhotoImage(plot_img)
        self.prediction_panel.configure(image=self.prediction_plot)
        self.prediction_panel.pack(side="top", fill="both", expand="yes")


myGUI = MyGUI()
