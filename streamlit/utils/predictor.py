"""
predictor
---------

Contains the trained model/models that will be used for
sound prediction. It runs in a separate thread monitoring
a queue from which it'll load the audio to predict
"""

import pickle
import threading
from pathlib import Path

from .communications import predictQueue, resultsQueue
from .qmodel import compile_model

# Directories
BASE_DIR = f"{Path(__file__).parent.resolve()}"
MODELS_DIR = "/../models/"
FEATURES_DIR = "/../features/"
WEIGHTS_FILENAME = "qiuqiangkong_b64_weights.tf"
MULTILABEL_FILENAME = "multiLabelBinarizer.pkl"
CLASSES_DICTIONARY_FILENAME = "classes_dict.pkl"


class Predictor(threading.Thread):
    feat_extractor = None
    model = None
    mlb = None
    classes_dict = None
    audio_type = None

    def __init__(self):
        """Loads trained model"""
        super(Predictor, self).__init__()

        # Load model from class
        print("Compile model: ", type(compile_model()))
        self.model = compile_model()
        self.model.load_weights(BASE_DIR + MODELS_DIR + WEIGHTS_FILENAME)

        # Load MultiLabelBinarizer
        with open(BASE_DIR + FEATURES_DIR + MULTILABEL_FILENAME, "rb") as mlb_file:
            self.mlb = pickle.load(mlb_file)
            print(f"Loaded clases: {self.mlb.classes_}")
        # Load classes dictionary
        with open(
            BASE_DIR + FEATURES_DIR + CLASSES_DICTIONARY_FILENAME, "rb"
        ) as classes_file:
            self.classes_dict = pickle.load(classes_file)
            print("Loaded classes names")

    def run(self):
        """Overwrites Thread run function. Waits for the queue and predicts"""
        while True:
            features = predictQueue.get()

            # Get prediction, add 1 dimension
            res = self.model.predict(
                features.reshape(1, features.shape[0], features.shape[1])
            )

            print(f"Prediction: {res}, res.argmax = {res.argmax()}")
            # Get translated result
            sound_names = self.classes_dict.get(self.mlb.classes_[res.argmax()])

            print(f"Sound name: {sound_names}")
            # Put it on the Queue for processing
            resultsQueue.put(sound_names)
