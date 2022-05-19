import logging
import logging.handlers
import queue

import pydub
from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    WebRtcMode,
    webrtc_streamer,
)

import streamlit as st

# Local imports
import utils.communications
from utils.predictor import Predictor
from utils.sound_utils import SoundHelper

# Introduction
st.title("Urban Sounds Detection")
st.image("./images/MadridSkyline.jpg")
st.markdown(
    "Hi! Welcome to my final project on [CoreCode](https://corecode.school/) Bootcamp, where I've been studying Machine Learning"
)
st.markdown(
    "In this case, I'll try to apply ML models to sounds captured from the streets of the city where I live, Madrid, and try to classify them. You can try it on yours, of course!"
)
st.markdown(
    "I've used [Google AudioSet](https://research.google.com/audioset/index.html) for training and selected 10 different categories: \n- Vehicle horn\n- Children playing\n- Dog\n- Jackhammer\n- Siren\n- Traffic noise\n- Subway\n- Footsteps\n- Chatter\n- Bird"
)


# Load inference model
@st.experimental_singleton
def get_predictor():
    return Predictor()


# Load sound helper
@st.experimental_singleton
def get_sound_helper():
    return SoundHelper()


# They will be loaded only once
predictor = get_predictor()
if "predictor" not in st.session_state:
    predictor.start()
    st.session_state["predictor"] = "running"
sound_helper = get_sound_helper()

# Get logger
logger = logging.getLogger(__name__)

# Create webrtc for audio only
webrtc_ctx = webrtc_streamer(
    key="Urban Sounds Detection",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True},
)

# Length of the chunks to be created in ms
sound_window_len = 5000
# Initialize sound object
sound_window_buffer = pydub.AudioSegment.empty()

# Display predicted class
tag = st.empty()

# Main Loop
while True:
    if webrtc_ctx.audio_receiver:
        try:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            logger.warning("Queue is empty. Abort.")
            break
        sound_chunk = pydub.AudioSegment.empty()

        for audio_frame in audio_frames:
            # Create sound chunk from received audio
            sound = pydub.AudioSegment(
                data=audio_frame.to_ndarray().tobytes(),
                sample_width=audio_frame.format.bytes,
                frame_rate=audio_frame.sample_rate,
                channels=len(audio_frame.layout.channels),
            )
            sound_chunk += sound

        if len(sound_chunk) > 0:
            sound_window_buffer += sound_chunk
            if len(sound_window_buffer) > sound_window_len:
                sound_window_buffer = sound_window_buffer[-sound_window_len:]
        # Save sound file when it reaches the specified length
        if sound_window_buffer.duration_seconds >= sound_window_len / 1000:
            # sound = sh.save_sound(sound_window_buffer)
            sound_helper.save_sound(sound_window_buffer)
            sound_window_buffer = pydub.AudioSegment.empty()
            # sound_window_buffer = sound_window_buffer.set_channels(1)  # Stereo to mono
            # filename = f"..\sounds\sound{i}.wav"
            # sound_window_buffer.export(filename, format="wav")
            # features = fe.extract_features(sound)
            # i += 1
            # Empty sound
            try:
                tag_read = utils.communications.predictQueue.get_nowait()
                print(tag_read)

                # Display in our web
                tag.header(tag_read)
            except queue.Empty:
                pass

    else:
        logger.warning("AudioReceiver is not set. Abort.")
        break
