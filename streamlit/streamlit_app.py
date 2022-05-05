import logging
import logging.handlers
import queue
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import List

import av
import matplotlib.pyplot as plt
import numpy as np
import pydub
from streamlit_webrtc import (AudioProcessorBase, ClientSettings, WebRtcMode,
                              webrtc_streamer)

import streamlit as st

st.title("Urban Sounds Detection")

logger = logging.getLogger(__name__)

webrtc_ctx = webrtc_streamer(
    key="Urban Sounds Detection",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    # rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True},
)

sound_window_len = 10000  # 10s chunks
sound_window_buffer = pydub.AudioSegment.empty()
i = 0

while True:
    if webrtc_ctx.audio_receiver:
        try:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            logger.warning("Queue is empty. Abort.")
            break
        sound_chunk = pydub.AudioSegment.empty()
        print("Audio frames: ", len(audio_frames))        
        for audio_frame in audio_frames:
            print("Sample width: ", audio_frame.format.bytes)
            sound = pydub.AudioSegment(
                data=audio_frame.to_ndarray().tobytes(),
                sample_width=audio_frame.format.bytes,
                frame_rate=audio_frame.sample_rate,
                channels=len(audio_frame.layout.channels),
            )
            print("Sound duration:", sound.duration_seconds)
            sound_chunk += sound
            print("Sound_chunk: ", sound_chunk.duration_seconds)
        print("For loop end")
        if len(sound_chunk) > 0:
            # No sound
            # if sound_window_buffer is None:
            #    print("Enter silent")
            #    sound_window_buffer = pydub.AudioSegment.silent(
            #        duration=sound_window_len
            #    )
            print("Enter if")
            sound_window_buffer += sound_chunk
            print("Current sound window buffer: ", sound_window_buffer.duration_seconds)
            if len(sound_window_buffer) > sound_window_len:
                print("Len Sound Buffer: ", len(sound_window_buffer))
                sound_window_buffer = sound_window_buffer[-sound_window_len:]
        if sound_window_buffer.duration_seconds >= 10:
            # Ref: https://own-search-and-study.xyz/2017/10/27/python%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E9%9F%B3%E5%A3%B0%E3%83%87%E3%83%BC%E3%82%BF%E3%81%8B%E3%82%89%E3%82%B9%E3%83%9A%E3%82%AF%E3%83%88%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%A0%E3%82%92%E4%BD%9C/
            sound_window_buffer = sound_window_buffer.set_channels(1)  # Stereo to mono
            print(
                "Exporting sound buffer. Duration: ",
                sound_window_buffer.duration_seconds,
            )
            sound_window_buffer.export(f"..\sounds\sound{i}.wav", format="wav")
            i += 1
            # Empty sound
            sound_window_buffer = pydub.AudioSegment.empty()
    else:
        logger.warning("AudioReceiver is not set. Abort.")
        break
