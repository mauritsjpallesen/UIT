import kivy
kivy.require('1.8.0')

import numpy as np

from kivymd.app import MDApp
from kivy.uix.label import Label
from kivy.network.urlrequest import UrlRequest
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.lang import Builder

from pitchDetection import getPrediction, getAverageFrequncy

import sounddevice as sd
from scipy.io.wavfile import write

Builder.load_file('uit.kv')




class Uit(Screen):

    def __init__(self, **kwargs):
        super(Uit, self).__init__(**kwargs)
        self.counter = 0
        self.recording = None
        self.modelEndpoint = "https://max-audio-classifier.codait-prod-41208c73af8fca213512856c7a09db52-0000.us-east.containers.appdomain.cloud/model/predict"

    def record(self, filename):
        print("we innit")
        sd.default.device = 'default'
        fs = 44100
        if self.counter % 2 == 0:
            seconds = 10

            self.recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
            self.counter += 1
            # sd.wait()
        else:
            sd.stop()
            write(filename + '.wav', fs, self.recording)
            print("success")



    def save_recording(self, data):
        print('i got', len(data))

    def predict(self, audioFile, confidenceThreshold):
        prediction = getPrediction(audioFile)
        avgFreq = getAverageFrequncy(prediction, confidenceThreshold)
        print(avgFreq)


class MyApp(MDApp):

    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.primary_hue = "800"

        self.sm = ScreenManager()
        self.sm.add_widget(Uit(name="name"))
        return self.sm

if __name__ == '__main__':
    MyApp().run()
