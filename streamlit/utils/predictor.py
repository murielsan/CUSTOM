import pickle
import threading
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer

from .communications import predictQueue, soundsQueue
# print(f"Path: {Path(__file__).parent.resolve()}")
# sys.path.append(Path(__file__).parent.resolve())
from .extract_features import FeatureExtractor

MODEL_FILENAME = "keras_audio.h5"
MULTILABEL_FILENAME = "multiLabelBinarizer.pkl"
THRESHOLD = 0.50
CLASSES_DICTIONARY = {
    53: "Vehicle horn, car horn, honking",
    68: "Children playing",
    71: "Dog",
    74: "Jackhammer",
    111: "Siren",
    308: "Traffic noise, roadway noise",
    327: "Subway, metro, underground",
    334: "Walk, footsteps",
    396: "Chatter",
    420: "Bird",
}


class Predictor(threading.Thread):
    feat_extractor = None
    model = None
    mlb = None

    def __init__(self):
        super(Predictor, self).__init__()
        # Load Feature extractor
        self.feat_extractor = FeatureExtractor()
        # Load model:
        self.model = tf.keras.models.load_model(
            f"{Path(__file__).parent.resolve()}/{MODEL_FILENAME}"
        )
        # Load MultiLabelBinarizer
        with open(
            f"{Path(__file__).parent.resolve()}/{MULTILABEL_FILENAME}", "rb"
        ) as mlb_file:
            self.mlb = pickle.load(mlb_file)

    def run(self):
        while True:
            features = self.feat_extractor.extract_features(soundsQueue.get())

            for x in features:
                # Predict
                res = self.model(x.reshape((1, 128, 1)))
                # THIS WAS MULTILABEL CASE
                # Filter to activate or not a neurone
                # res = np.vectorize(lambda x: 1 if x == THRESHOLD else 0)(res)
                # Transform results into labels
                # sounds = self.mlb.inverse_transform(res)
                # Transform labels into names
                # sound_names = [CLASSES_DICTIONARY.get(s) for s in sounds[0]]

                # MULTICLASS
                sound_names = CLASSES_DICTIONARY.get(self.mlb.classes_[res.numpy().argmax()])

                predictQueue.put(sound_names)
