import logging
import logging.handlers
import queue

import pydub
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import streamlit as st

# Local imports
import utils.communications
from utils.extract_features import FeatureExtractor
from utils.predictor import Predictor
from utils.sound_utils import SoundHelper

##### INITIALIZATIONS ####
if "audio_type" not in st.session_state:
    st.session_state["audio_type"] = "stream"  # wav/stream
    st.session_state["sound_window_len"] = 10000  # 10s window

# Load features extractor
@st.experimental_singleton
def get_features_extractor(at):
    return FeatureExtractor(audio_type=at)


# Load inference model
@st.experimental_singleton
def get_predictor():
    return Predictor()


# Load sound helper
@st.experimental_singleton
def get_sound_helper(audio_type):
    return SoundHelper(audio_type)


# Get logger
logger = logging.getLogger(__name__)


# Web RTC component status
def get_webrtc_status(comp):
    if comp.state.playing:
        return "Recording..."
    else:
        return "Stopped"


#### MAIN CONTENT ####
# Introduction
st.title("Urban Sounds Detection")
st.image("./images/MadridSkyline.jpg")
st.markdown(
    "Hi! Welcome to my final project on [CoreCode](https://corecode.school/) "
    "Bootcamp, where I've been studying Machine Learning"
)
st.markdown(
    "In this case, I'll try to apply ML models to sounds captured from the "
    "streets of the city where I live, Madrid, and try to classify them. "
    "You can try it on yours, of course!"
)
st.markdown(
    "I've used [Google AudioSet](https://research.google.com/audioset/index.html)"
    " for training and selected 10 different categories: \n- Vehicle horn\n- "
    "Children playing\n- Dog\n- Jackhammer\n- Siren\n- Traffic noise\n- Subway"
    "\n- Footsteps\n- Chatter\n- Bird"
)


# They will be loaded only once
if "initialization" not in st.session_state:
    predictor = get_predictor()
    predictor.start()

    features_extractor = get_features_extractor(st.session_state.audio_type)
    features_extractor.start()

    st.session_state["sound_helper"] = get_sound_helper(st.session_state.audio_type)
    st.session_state["initialization"] = "done"


# Create webrtc for audio only
webrtc_ctx = webrtc_streamer(
    key="Urban Sounds Detection",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    # For web deployment
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True},
)

# Display predicted class
recorder_state = st.empty()
tag = st.empty()

# Main Loop
while True:
    recorder_state.text(get_webrtc_status(webrtc_ctx))
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
            try:
                sound_window_buffer += sound_chunk
            except NameError:
                sound_window_buffer = pydub.AudioSegment.empty()
                sound_window_buffer += sound_chunk

            if len(sound_window_buffer) >= st.session_state.sound_window_len:
                print("Got 10 seconds, passing audio")
                st.session_state.sound_helper.queue_sound(sound_window_buffer)

                # Empty buffer
                sound_window_buffer = pydub.AudioSegment.empty()
            try:
                tag_read = utils.communications.resultsQueue.get_nowait()
                print(tag_read)

                # Display in our web
                tag.header(tag_read)
            except queue.Empty:
                pass

    else:
        logger.warning("AudioReceiver is not set. Abort.")
        break
