import kivy
kivy.require('1.8.0')

from kivymd.app import MDApp
from kivy.uix.label import Label
from kivy.network.urlrequest import UrlRequest
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.lang import Builder
from audiostream import get_input

from requests_toolbelt import MultipartEncoder

Builder.load_file('uit.kv')




class Uit(Screen):

    def __init__(self, **kwargs):
        super(Uit, self).__init__(**kwargs)
        self.counter = 0
        self.modelEndpoint = "https://max-audio-classifier.codait-prod-41208c73af8fca213512856c7a09db52-0000.us-east.containers.appdomain.cloud/model/predict"

    def record(self, filename):
        if self.counter % 2 == 0:
            mic = get_input(callback=save_recording)
            mic.start()
            self.counter += 1
        else:
            self.rec.record = False


    def save_recording(self, data):
        print('i got', len(data))

    def predictSuccess(self, *args):
        print(args)
        print("success")

    def predictFailure(self, *args):
        print(args)
        print("failure")

    def predictError(self, *args):
        print(args)
        print("error")

    def predict(self, audioFile="/home/maurits/Git/UIT/kivy/maxAudioClassifierTest/audios/mugTap.wav"):
        files = {"file": open(audioFile, "rb")}

        with open(audioFile, "rb") as fobj:
            payload = MultipartEncoder(
                        fields={
                            "audio": (
                                "audio.wav",
                                fobj,
                                "audio/wav"
                            )
                        }
                    )

            headers = {'Accept': 'application/json'}
            req = UrlRequest(
                self.modelEndpoint,
                on_success = self.predictSuccess,
                on_failure = self.predictFailure,
                on_error = self.predictError,
                req_headers = headers,
                req_body=payload
            )


class MyApp(MDApp):

    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.primary_hue = "800"

        self.sm = ScreenManager()
        self.sm.add_widget(Uit(name="name"))
        return self.sm

if __name__ == '__main__':
    MyApp().run()
