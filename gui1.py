import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pyaudio
import os
import wave
import librosa
import numpy as np
from sys import byteorder
from array import array
from struct import pack
from keras.models import load_model

# Load the voice gender detection model
gender_model = load_model('model.h5')  # Replace 'your_voice_gender_model.h5' with your actual model file

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000
SILENCE = 30

class VoiceGenderRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Gender Recognition App")
        self.root.geometry('800x600')
        self.root.configure(background='#CDCDCD')

        self.label1 = tk.Label(root, background="#CDCDCD", font=('arial', 15, "bold"))
        self.label2 = tk.Label(root, background="#CDCDCD", font=('arial', 15, 'bold'))
        self.label1.pack(side="bottom", expand=True)
        self.label2.pack(side="bottom", expand=True)

        self.upload_button = tk.Button(root, text="Upload", command=self.upload_voice_file, padx=10, pady=5)
        self.upload_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
        self.upload_button.pack(side='bottom', pady=10)

        self.detect_button = tk.Button(root, text="Detect", command=self.detect_gender, padx=10, pady=5)
        self.detect_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
        self.detect_button.pack(side='bottom', pady=10)

    def is_silent(self, snd_data):
        return max(snd_data) < THRESHOLD

    def record(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=1, rate=RATE,
                        input=True, output=True,
                        frames_per_buffer=CHUNK_SIZE)

        num_silent = 0
        snd_started = False

        r = array('h')

        while 1:
            snd_data = array('h', stream.read(CHUNK_SIZE))
            if byteorder == 'big':
                snd_data.byteswap()
            r.extend(snd_data)

            silent = self.is_silent(snd_data)

            if silent and snd_started:
                num_silent += 1
            elif not silent and not snd_started:
                snd_started = True

            if snd_started and num_silent > SILENCE:
                break

        sample_width = p.get_sample_size(FORMAT)
        stream.stop_stream()
        stream.close()
        p.terminate()

        r = self.normalize(r)
        r = self.trim(r)
        r = self.add_silence(r, 0.5)
        return sample_width, r

    def normalize(self, snd_data):
        MAXIMUM = 16384
        times = float(MAXIMUM) / max(abs(i) for i in snd_data)

        normalized_data = array('h')
        for i in snd_data:
            normalized_data.append(int(i * times))
        return normalized_data

    def trim(self, snd_data):
        def _trim(snd_data):
            snd_started = False
            trimmed_data = array('h')

            for i in snd_data:
                if not snd_started and abs(i) > THRESHOLD:
                    snd_started = True
                    trimmed_data.append(i)

                elif snd_started:
                    trimmed_data.append(i)
            return trimmed_data

        # Trim to the left
        snd_data = _trim(snd_data)

        # Trim to the right
        snd_data.reverse()
        snd_data = _trim(snd_data)
        snd_data.reverse()
        return snd_data

    def add_silence(self, snd_data, seconds):
        silence_data = array('h', [0 for i in range(int(seconds * RATE))])
        silence_data.extend(snd_data)
        silence_data.extend([0 for i in range(int(seconds * RATE))])
        return silence_data

    def extract_feature_from_data(self, snd_data):
        X, sample_rate = np.array(snd_data), RATE
        result = np.array([])

        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))

        return result

    def detect_gender(self):
        try:
            sample_width, data = self.record()
            features = self.extract_feature_from_data(data).reshape(1, -1)
            male_prob = gender_model.predict(features)[0][0]
            female_prob = 1 - male_prob
            gender = "male" if male_prob > female_prob else "female"

            messagebox.showinfo("Result", f"Detected Gender: {gender}\nMale Probability: {male_prob*100:.2f}%\nFemale Probability: {female_prob*100:.2f}%")
            self.label1.configure(foreground="#011638", text=f"Gender: {gender}")
            self.label2.configure(foreground="#011638", text=f"Male Probability: {male_prob*100:.2f}%\nFemale Probability: {female_prob*100:.2f}%")
        except Exception as e:
            print(e)

    def upload_voice_file(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3")])
            if file_path:
                features = self.extract_feature_from_file(file_path).reshape(1, -1)
                male_prob = gender_model.predict(features)[0][0]
                female_prob = 1 - male_prob
                gender = "male" if male_prob > female_prob else "female"

                messagebox.showinfo("Result", f"Detected Gender: {gender}\nMale Probability: {male_prob*100:.2f}%\nFemale Probability: {female_prob*100:.2f}%")
                self.label1.configure(foreground="#011638", text=f"Gender: {gender}")
                self.label2.configure(foreground="#011638", text=f"Male Probability: {male_prob*100:.2f}%\nFemale Probability: {female_prob*100:.2f}%")
        except Exception as e:
            print(e)

    def extract_feature_from_file(self, file_path):
        X, sample_rate = librosa.load(file_path)
        result = np.array([])

        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))

        return result

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceGenderRecognitionApp(root)
    root.mainloop()

