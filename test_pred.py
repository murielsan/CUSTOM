import sys
from importlib.resources import files
from pathlib import PurePath

paths = [
    str(PurePath(sys.path[0]).joinpath("./models/research/audioset/vggish")),
    str(PurePath(sys.path[0]).joinpath("./streamlit/utils")),
]

sys.path += paths

import glob
import os
import pickle

import numpy as np
import soundfile
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
from extract_features import FeatureExtractor
from pydub import AudioSegment
from qmodel import compile_model

# Load MultiLabelBinarizer
with open("./data_preparation/features/multiLabelBinarizer.pkl", "rb") as mlb_file:
    mlb = pickle.load(mlb_file)
# Load classes dictionary
with open("./data_preparation/features/classes_dict.pkl", "rb") as classes_file:
    classes_dict = pickle.load(classes_file)


sounds_dir = "./test_sounds"

# Load model
model = compile_model()
model.load_weights("./data_preparation/models/qiuqiangkong_b64_weights.tf")
fe = FeatureExtractor()

for current_file in glob.glob(f"{sounds_dir}/*.wav"):

    print(f"---- File {current_file} -----")
    # Soundfile read, compatible with VGGish
    sound0, sr = soundfile.read(current_file)
    print(sound0)

    # pydub read
    sound1 = AudioSegment.from_file(
        current_file,
        format="wav",
        frame_rate=48000,
        channels=2,
        sample_width=2,
    )
    # sound1 = sound1.set_channels(1)  # to mono audio
    print(f"Samples = {type(sound1.get_array_of_samples())}")
    y = np.array(sound1.get_array_of_samples()).astype(np.float32)
    y = y / 32768.0
    print(f"sound1: {y.shape}, {y}")

    features = fe.extract_features(current_file)

    # Get prediction, add 1 dimension
    print("From wav file")
    res = model.predict(features.reshape(1, features.shape[0], features.shape[1]))

    print(res)
    print(classes_dict.get(mlb.classes_[res.argmax()]))

    print("From stream")
    features = fe.extract_features_from_stream(sound1)

    # Get prediction, add 1 dimension
    res = model.predict(features.reshape(1, features.shape[0], features.shape[1]))
    print(res)
    print(classes_dict.get(mlb.classes_[res.argmax()]))
