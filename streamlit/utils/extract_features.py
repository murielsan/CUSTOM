import sys
from pathlib import PurePath

import numpy as np
from pydub import AudioSegment

vggish_path = PurePath(sys.path[0]).parent.joinpath(
    "../models/research/audioset/vggish"
    # "C:/Users/jm250119/Documents/CoreCode/CUSTOM/models/research/audioset/vggish"
)
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


class FeatureExtractor:
    pproc = None
    sess = None
    features_tensor = None
    embedding_tensor = None

    # Dummy file for CUDA preload
    dummy_wav = None

    def __init__(self):
        # Prepare a postprocessor to munge the model embeddings.
        self.pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

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

    def extract_features(self, wav_file):

        examples_batch = vggish_input.wavfile_to_examples(wav_file)

        # Run inference and postprocessing.
        [embedding_batch] = self.sess.run(
            [self.embedding_tensor], feed_dict={self.features_tensor: examples_batch}
        )
        postprocessed_batch = self.pproc.postprocess(embedding_batch)

        return postprocessed_batch

    def extract_features_from_stream(self, audio: AudioSegment):

        frame_rate = audio.frame_rate
        audio = (
            np.array(
                # audio.set_frame_rate(44100).split_to_mono()[0].get_array_of_samples()
                audio.split_to_mono()[0].get_array_of_samples()
            ).astype(np.float32)
            / 32768.0
        )
        print(f"Audio shape: {audio.shape}")
        examples_batch = vggish_input.waveform_to_examples(audio, frame_rate)

        # Run inference and postprocessing.
        [embedding_batch] = self.sess.run(
            [self.embedding_tensor], feed_dict={self.features_tensor: examples_batch}
        )
        postprocessed_batch = self.pproc.postprocess(embedding_batch)

        return postprocessed_batch

    def init_cuda(self):
        self.dummy_wav = f"{str(PurePath(sys.path[0]).parent)}/utils/dummy.wav"
        # self.dummy_wav = (f"C:/Users/jm250119/Documents/CoreCode/CUSTOM/streamlit/utils/dummy.wav")

        self.sess.run(
            [self.embedding_tensor],
            feed_dict={
                self.features_tensor: vggish_input.wavfile_to_examples(self.dummy_wav)
            },
        )
