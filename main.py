import kivy
kivy.require('1.8.0')

import numpy as np
import os

import threading

from kivy.clock import Clock, mainthread
from kivy.metrics import dp
from kivymd.app import MDApp
from kivy.uix.label import Label
from kivy.network.urlrequest import UrlRequest
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.lang import Builder
from kivymd.uix.dialog import MDDialog
from kivymd.uix.slider import MDSlider
from kivymd.uix.filemanager import MDFileManager
from kivy.core.window import Window
from kivymd.uix.spinner import MDSpinner
import ml

from pitchDetection import getAverageFrequency

import sounddevice as sd
from scipy.io.wavfile import write

Builder.load_file('uit.kv')

class Uit(Screen):

    def __init__(self, **kwargs):
        super(Uit, self).__init__(**kwargs)

        X_train, X_test, y_train, y_test, encoder = ml.getTrainAndTestData(0.1)
        clf = ml.getTrainedSvmModel(X_train, y_train)
        self.svm = clf
        self.svmEncoder = encoder

        self.spinner = MDSpinner(
            size_hint=(None, None),
            size=(dp(120), dp(120)),
            pos_hint={'center_x': .5, 'center_y': .5},
            active=False,
            palette=[
                [0.3568627450980392, 0.3215686274509804, 0.8666666666666667, 1],
                [0.8862745098039215, 0.36470588235294116, 0.592156862745098, 1],
                [0.8784313725490196, 0.9058823529411765, 0.40784313725490196, 1],
                [0.28627450980392155, 0.8431372549019608, 0.596078431372549, 1],
            ]
        )
        self.add_widget(self.spinner)
        self.filePath = ""
        self.threshhold = 0
        self.avgFreq = 0
        self.modelEndpoint = "https://max-audio-classifier.codait-prod-41208c73af8fca213512856c7a09db52-0000.us-east.containers.appdomain.cloud/model/predict"

    def record(self, filename):
        fs = 44100
        seconds = 3
        self.filePath = './audio/' + filename + '.wav'
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write(self.filePath, fs, recording)

    def set_slider(self, data):
        self.threshhold = data/100

    def set_path(self, path):
        self.filePath = path

    def files(self):
        path = '/audio'  # path to the directory that will be opened in the file manager
        file_manager = MDFileManager(
            exit_manager=self.exit_manager,  # function called when the user reaches directory tree root
            select_path=self.select_path,  # function called when selecting a file/directory
        )
        file_manager.show(path)

    def select_path(self, path):
        self.filePath = path

    def exit_manager(self, *args):
        self.manager_open = False
        self.file_manager.close()

    def predict(self):
        if self.filePath == "":
            dialog = MDDialog(title='No file',
              text='Please record first')
            dialog.open()
        else:
            features = ml.extractFeaturesFromFile(self.filePath, 0.4, 0.25)
            if (features == None):
                MDDialog(title='Unable to find sound').open()
                return

            print(features)
            prediction = self.svm.predict([features])[0]
            label = self.svmEncoder.inverse_transform([prediction])
            dialog = MDDialog(title='Result', text=str(label))
            dialog.open()



class MyApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)
        self.instance = Uit(name="Uit")
        self.manager_open = False
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
            ext = ['.wav']
        )

    def file_manager_open(self):
        cwd = os.getcwd()
        self.file_manager.show(cwd)  # output manager to the screen
        self.manager_open = True

    def select_path(self, path):
        self.instance.set_path(path)
        self.manager_open = False
        self.file_manager.close()

    def exit_manager(self, *args):
        self.manager_open = False
        self.file_manager.close()

    def events(self, instance, keyboard, keycode, text, modifiers):
        '''Called when buttons are pressed on the mobile device.'''

        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()
        return True

    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.primary_hue = "800"

        self.sm = ScreenManager()
        self.sm.add_widget(self.instance)
        return self.sm

if __name__ == '__main__':
    MyApp().run()
