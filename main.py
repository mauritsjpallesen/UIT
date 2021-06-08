import kivy
kivy.require('1.8.0')

import numpy as np
import os

import threading
import hashlib

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

from featureExtraction import extractFeaturesFromFile

from ml import getTrainAndTestData, getTrainedModel, predict
# from deepLearning import getTrainAndTestData, getTrainedModel


import sounddevice as sd
from scipy.io.wavfile import write

Builder.load_file('uit.kv')
Builder.load_file('login.kv')

class Login(Screen):
    def __init__(self, **kwargs):
        super(Login, self).__init__(**kwargs)

        self.filePath = ""
        X_train, X_test, y_train, y_test, encoder, scaler = getTrainAndTestData(0.01)
        model = getTrainedModel(X_train, y_train)
        self.model = model
        self.modelEncoder = encoder
        self.modelScaler = scaler
        self.local_storage = {
        "mau" : ['66d6fa803037211d7c0e40499f951c057a5c0329e4dfe528d8fa824bbca17624', '89d99fc4d6310c40934255843889c641797eb4e3a3f3cf659e06bef528161f13'],
        "anders" : ['17b86905af72dd8265d38535c56c7c91502cfeb6586d3df0dc3e78392ff82ecb', 'df1d29b186fac0f8d3b2169068c535965c1df85ce27c47420821111d1ed2b357'],
        "nathalia" : ['fc6127479cd0c90e7781e37ebb1d9b496664cd195319cc1c4c52d4f79ba08fab', '1939be098a57888bb5b8647842c2873e9527724cd29e39ae621f7106ff2a6b0e'],
        "mads" : ['cd484bbf277ebd0c0be0ce0690fee09a0e32f1f8add111c9636a7fc720017762', '256eac32fa3df1cf17aefb6533dfd06d20c86508f423cd87e5d88b10445b2c92']}

        #mau = erdetmindreng + ear
        #anders = lau + trunk
        #nathalia = krøigaard + backleg
        #mads = møn + upperbody

    def login(self, username, password):
        prediction = predict(self.model, self.modelScaler, self.modelEncoder, self.filePath)

        if self.local_storage[str(username)] == [hashlib.blake2s(password.encode()).hexdigest(), hashlib.blake2s(prediction.lower().encode()).hexdigest()]:
            self.manager.transition = SlideTransition(direction="left")
            self.manager.current = self.manager.next()
        else:
            dialog = MDDialog(title='Authentication failed',
              text='Please try again')
            dialog.open()

    def set_path(self, path):
        self.filePath = path

class Uit(Screen):

    def __init__(self, **kwargs):
        super(Uit, self).__init__(**kwargs)

        X_train, X_test, y_train, y_test, encoder, scaler = getTrainAndTestData(0.01)
        model = getTrainedModel(X_train, y_train)
        self.model = model
        self.modelEncoder = encoder
        self.modelScaler = scaler

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

    def predict(self):
        if self.filePath == "":
            dialog = MDDialog(title='No file',
              text='Please record first')
            dialog.open()
        else:
            predictionLabel = predict(self.model, self.modelScaler, self.modelEncoder, self.filePath)
            if (predictionLabel == None):
                MDDialog(title='Unable to find sound').open()
                return

            dialog = MDDialog(title='Result', text=predictionLabel)
            dialog.open()

class MyApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)
        self.sm = ScreenManager()
        self.instance = Uit(name="Uit")
        self.login = Login(name="Login")
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
        self.sm.current_screen.set_path(path)
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

        self.sm.add_widget(self.login)
        self.sm.add_widget(self.instance)
        return self.sm

if __name__ == '__main__':
    MyApp().run()
