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


# A GUI to test a saved network
class MyGUI:
    def __init__(self):
        self.root = tk.Tk()

        self.spectrogram_image = None
        self.prediction_plot = None
        self.audio_file = None
        self.model = None
        self.prediction_revealed = True

        self.init_model()

        self.frame = tk.Frame(self.root)
        self.frame.pack(side="top", fill="both", expand=True)

        self.menubar = tk.Menu(self.root)
        self.menubar.add_command(label="Quit", command=self.root.quit)
        self.menubar.add_command(label="Select audio", command=self.selectAudioFile)
        self.menubar.add_command(label="Play current audio", command=self.playAudioFile)

        # display the menu
        self.root.config(menu=self.menubar)

        # Label which displays the category of the loaded audio file
        self.label_audiofile = tk.Label(self.root, text="", font="Helvetica 16 bold")
        self.label_audiofile.pack(pady=3, in_=self.frame)

        # Display the spectrogram of the loaded audio file
        self.spectrogram_panel = tk.Label(self.root, image=self.spectrogram_image)
        self.spectrogram_panel.pack(pady=3, in_=self.frame)

        # A button used to show or hide the prediction
        self.button_reveal = tk.Button(self.root, text="REVEAL/HIDE", font="Helvetica 16 bold",
                                       command=self.reveal_or_hide_prediction)
        self.button_reveal.pack(pady=8, in_=self.frame, side="bottom")

        self.label_prediction = tk.Label(self.root, text="", font="Helvetica 16 bold")
        self.label_prediction.pack(pady=3, in_=self.frame, side="bottom")

        self.prediction_panel = tk.Label(self.root, image=self.prediction_plot)
        self.prediction_panel.pack(pady=3, in_=self.frame, side="bottom")

        self.root.mainloop()

    def init_model(self):
        input_shape = (92, 193, 1)
        # Load the model for the first time only
        self.model = basic_cnn(NB_CLASSES, input_shape)
        self.model.load_weights(MODELS_DIR + 'best-model.hdf5')

    def show_widget(self, widget):
        widget.lift(self.frame)

    def hide_widget(self, widget):
        widget.lower(self.frame)

    def reveal_or_hide_prediction(self):
        if self.prediction_revealed:
            self.hide_widget(self.spectrogram_panel)
            self.hide_widget(self.label_audiofile)
            if self.label_prediction is not None:
                    self.hide_widget(self.label_prediction)
            if self.prediction_panel is not None:
                self.hide_widget(self.prediction_panel)
        else:
            self.show_widget(self.spectrogram_panel)
            self.show_widget(self.label_audiofile)
            if self.label_prediction is not None:
                self.show_widget(self.label_prediction)
            if self.prediction_panel is not None:
                self.show_widget(self.prediction_panel)
        self.prediction_revealed = not self.prediction_revealed

    def playAudioFile(self):
        os.system("start " + self.audio_file)

    def selectAudioFile(self):
        self.root.filename = filedialog.askopenfilename(
            initialdir="C:\\Users\\NNED\\PycharmProjects\\urban_sound_classification\\data\\audio\\fold1",
            title="Select file", filetypes=(("wav files", "*.wav"), ("all files", "*.*")))
        if self.root.filename is None or self.root.filename is '':
            return
        self.audio_file = self.root.filename

        category = self.audio_file.split("-")[-3]
        self.label_audiofile['text'] = 'File : ' + self.audio_file.split('/')[-1] + '\nCategory : ' + convert_category_to_name(category).replace('_', ' ')
        # Load the choosen audio file
        signal, sample_rate = librosa.load(self.root.filename, sr=None)
        # Generate the audio file's spectrogram
        spectrogram = calculate_spectrogram(signal=signal, sample_rate=sample_rate)
        # Normalize it for displaying purposes
        normalized_spectrogram = normalize_data(spectrogram.copy())
        normalized_spectrogram = np.flip(normalized_spectrogram, 0)
        # Create an Image from the spectrogram's array
        pilImage = Image.fromarray(np.uint8(normalized_spectrogram * 255), 'L')
        # Let's give it some colors
        colored_Image = ImageOps.colorize(image=pilImage, black='#000000', mid='#cc5015', white='#ffeeaa')

        width, height = colored_Image.size
        maxsize = (width * 3, height * 3)
        # It's better to resize it, otherwise it's too small
        resized_image = colored_Image.resize(maxsize, Image.BOX)

        self.spectrogram_image = ImageTk.PhotoImage(resized_image)
        self.spectrogram_panel.configure(image=self.spectrogram_image)
        self.spectrogram_panel.pack(side="top", fill="both", expand="yes", in_=self.frame)

        data = np.reshape(spectrogram, (1, spectrogram.shape[-2], spectrogram.shape[-1], 1))

        y_probabilities = self.model.predict(data, verbose=2)[0]
        y_predicted_classe = y_probabilities.argmax(axis=-1)
        prediction = 'Prediction : ' + convert_category_to_name(y_predicted_classe)

        self.label_prediction['text'] = prediction.replace('_', ' ')

        colors = cm.plasma(y_probabilities / float(max(y_probabilities)))
        plot = plt.scatter(y_probabilities, y_probabilities, c=y_probabilities, cmap='plasma')
        plt.clf()
        plt.colorbar(plot)
        plt.bar(CLASSES.values(), y_probabilities, color=colors)
        plt.xticks(rotation=60)
        plt.subplots_adjust(bottom=0.25)

        plot_path = 'plot.png'
        plt.savefig(plot_path)

        plot_img = Image.open(plot_path)
        self.prediction_plot = ImageTk.PhotoImage(plot_img)
        self.prediction_panel.configure(image=self.prediction_plot)
        self.prediction_panel.pack(side="top", fill="both", expand="yes", in_=self.frame)


myGUI = MyGUI()
