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

# Directories
BASE_DIR = f"{Path(__file__).parent.resolve()}"
MODELS_DIR = "/../../data_preparation/models/"
FEATURES_DIR = "/../../data_preparation/features/"
MODEL_FILENAME = "keras_audio.h5"
MULTILABEL_FILENAME = "multiLabelBinarizer.pkl"
CLASSES_DICTIONARY_FILENAME = "classes_dict.pkl"


class Predictor(threading.Thread):
    feat_extractor = None
    model = None
    mlb = None
    classes_dict = None

    def __init__(self):
        super(Predictor, self).__init__()
        # Load Feature extractor
        self.feat_extractor = FeatureExtractor()

        # Load model:
        self.model = tf.keras.models.load_model(BASE_DIR + MODELS_DIR + MODEL_FILENAME)
        # Load MultiLabelBinarizer
        with open(BASE_DIR + FEATURES_DIR + MULTILABEL_FILENAME, "rb") as mlb_file:
            self.mlb = pickle.load(mlb_file)
        # Load classes dictionary
        with open(
            BASE_DIR + FEATURES_DIR + CLASSES_DICTIONARY_FILENAME, "rb"
        ) as classes_file:
            self.classes_dict = pickle.load(classes_file)

    def run(self):
        while True:
            features = self.feat_extractor.extract_features(soundsQueue.get())

            for x in features:
                # Predict
                res = self.model(x.reshape((1, 128, 1)))
                # THIS WAS FOR MULTILABEL APPROACH
                # Filter to activate or not a neurone
                # res = np.vectorize(lambda x: 1 if x == THRESHOLD else 0)(res)
                # Transform results into labels
                # sounds = self.mlb.inverse_transform(res)
                # Transform labels into names
                # sound_names = [CLASSES_DICTIONARY.get(s) for s in sounds[0]]

                # MULTICLASS
                sound_names = self.classes_dict.get(
                    self.mlb.classes_[res.numpy().argmax()]
                )

                predictQueue.put(sound_names)
