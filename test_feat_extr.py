import sys
from importlib.resources import files
from pathlib import PurePath

paths = [
    str(PurePath(sys.path[0]).joinpath("./models/research/audioset/vggish")),
    str(PurePath(sys.path[0]).joinpath("./streamlit/utils")),
]

sys.path += paths

import os
import pickle

from extract_features import FeatureExtractor

sounds_dir = "./test_sounds/frames"

# Load Feature extractor
fe = FeatureExtractor(postprocess=False)

for current_file in os.listdir(sounds_dir):

    print(f"---- File {current_file} -----")
    features = fe.extract_features(f"{sounds_dir}/{current_file}")

    if features.shape[0] > 1:
        for f in features:
            print(f)
    else:
        print(features)
