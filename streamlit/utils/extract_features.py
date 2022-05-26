"""
extract features
----------------

Contains the functions needed to extract features or audio embeddings
from audio sources using VGGISH neural network created by Google
"""
import sys
import threading
from pathlib import PurePath

import numpy as np
from pydub import AudioSegment

from .communications import predictQueue, soundsQueue

MODE = "RELEASE"
# Need to change directories for my testing
if MODE == "TEST":
    vggish_path = PurePath(sys.path[0]).parent.joinpath(
        "C:/Users/jm250119/Documents/CoreCode/CUSTOM/streamlit/models/vggish"
    )
    DUMMY_SOUND_FILE = (
        "C:/Users/jm250119/Documents/CoreCode/CUSTOM/streamlit/utils/dummy.wav"
    )
else:
    vggish_path = PurePath(sys.path[0]).parent.joinpath("./models/vggish")
    DUMMY_SOUND_FILE = f"{str(PurePath(sys.path[0]).parent)}/utils/dummy.wav"
sys.path.append(str(vggish_path))

import tensorflow.compat.v1 as tf
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

flags = tf.app.flags

flags.DEFINE_string(
    "wav_file",
    None,
    "Path to a wav file. Should contain signed 16-bit PCM samples. "
    "If none is provided, a synthetic sound is used.",
)

flags.DEFINE_string(
    "checkpoint",
    f"{vggish_path}/vggish_model.ckpt",
    "Path to the VGGish checkpoint file.",
)

flags.DEFINE_string(
    "pca_params",
    f"{vggish_path}/vggish_pca_params.npz",
    "Path to the VGGish PCA parameters file.",
)

flags.DEFINE_string(
    "tfrecord_file", None, "Path to a TFRecord file where embeddings will be written."
)

FLAGS = flags.FLAGS


class FeatureExtractor(threading.Thread):
    pproc = None
    sess = None
    features_tensor = None
    embedding_tensor = None
    audio_type = None

    # Dummy file for CUDA preload
    dummy_wav = None

    def __init__(self, postprocess=True, audio_type="wav"):
        """Loads tensorflow, which takes some time"""
        super(FeatureExtractor, self).__init__()
        self.audio_type = audio_type

        # Prepare a postprocessor to munge the model embeddings.
        if postprocess:
            self.pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)
            self.postprocess = True
        else:
            self.postprocess = False

        with tf.Graph().as_default():
            # Try to reduce GPU memory usage
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            # Define the model in inference mode, load the checkpoint, and
            # locate input and output tensors.
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(self.sess, FLAGS.checkpoint)
            self.features_tensor = self.sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME
            )
            self.embedding_tensor = self.sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME
            )
        self.init_cuda()

    def run(self):
        if self.audio_type == "stream":
            while True:
                # Extract features from stream
                features = self.extract_features_from_stream(soundsQueue.get())
                print(f"Features: \n{features}")
                predictQueue.put(features)
        else:
            while True:
                # Extract features from file
                features = self.extract_features(soundsQueue.get())
                print(f"Features: \n{features}")
                predictQueue.put(features)

    def extract_features(self, wav_file):
        """Extract features from a wav sound file"""

        examples_batch = vggish_input.wavfile_to_examples(wav_file)

        # Run inference and postprocessing.
        [embedding_batch] = self.sess.run(
            [self.embedding_tensor], feed_dict={self.features_tensor: examples_batch}
        )
        print(f"Embedding batch: \n{embedding_batch}")
        if not self.postprocess:
            return embedding_batch
        else:
            return self.pproc.postprocess(embedding_batch)

    def extract_features_from_stream(self, audio: AudioSegment):
        """Extract features using pydub library -> AudioSegment"""
        print("Received streaming for feature extraction")
        audio_array = (
            np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        )
        print(f"Audio shape: {audio.channels}, Frame rate {audio.frame_rate}")

        examples_batch = vggish_input.waveform_to_examples(
            audio_array, audio.frame_rate
        )

        # Run inference and postprocessing.
        [embedding_batch] = self.sess.run(
            [self.embedding_tensor], feed_dict={self.features_tensor: examples_batch}
        )

        print(f"Embedding batch: \n{embedding_batch}")
        if not self.postprocess:
            return embedding_batch
        else:
            return self.pproc.postprocess(embedding_batch)

    def init_cuda(self):
        """Preload cuda environment to reduce time"""
        self.dummy_wav = DUMMY_SOUND_FILE

        self.sess.run(
            [self.embedding_tensor],
            feed_dict={
                self.features_tensor: vggish_input.wavfile_to_examples(self.dummy_wav)
            },
        )
