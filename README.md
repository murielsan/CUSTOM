# C.U.S.T.O.M.

![Madrid Skyline](images/MadridSkyline.jpg)
C.U.S.T.O.M. stands for Classifying Urban Sounds Taken On Madrid. It's a machine learning audio classifying proyect. Basically, I'm building a program that will be able to capture environmental sounds and classify them into one of the 10 labels our trained model will be able to identify.

## The dataset
We'll be using Google AudioSet dataset,[AudioSet: An ontology and human-labelled dataset for audio events](https://creativecommons.org/licenses/by-sa/4.0/) for training the different models. Also we'll use their pretrained models, [CNN Architectures for Large-Scale Audio Classification](https://research.google.com/pubs/pub45611.html).

Google's Audioset contains thousands of 10s audios (extracted from youtube videos) and 527 different classes provided as tensorflow records. After loading the balanced set records into variables and analyzing the data, these are the classes with the most number of samples:

![top 30 classes](images/top30classes.png)

### Classes selected
We've selected the following 10 classes to be identified by our model:

- :speech_balloon:Speech
- :guitar:Musical instrument
- :car:Car
- :dog:Dog
- :children_crossing:Child speech, kid speaking
- :train:Rail transport
- :ambulance:Siren
- :loudspeaker:Vehicle horn, car horn, honking
- :hammer:Jackhammer
- :rat:Pigeon, dove

These are common sounds you could hear when you walk around Madrid. We'll be training our models to identify these sounds.

## [API](api)
We've deployed our predictor to [Heroku](https://custom-corecode-api.herokuapp.com/) and created a [web test](https://murielsan.github.io/) so you can try it. The API contains only one entry point, through a post command:

| ENTRY POINT | TYPE | PARAMETERS | RETURNS          |
| :---------- | ---- | ---------- | ---------------- |
| predict     | POST | audio file | sound prediction |

It should process almost any sound file (anything decoded by [ffmpeg](https://ffmpeg.org/)), but ogg or wav are recommended.

The code of the web page is also available [here](https://github.com/murielsan/murielsan.github.io).

## [Streamlit application](streamlit)

With our model trained, we created a web application using [Streamlit](https://streamlit.io). The application will capture audio using your computer or mobile audio and will predict the sound being recorded for every 10seconds of audio.

The module used for audio capture [streamlit_webrtc](https://github.com/whitphx/streamlit-webrtc). It provides the sound as an AudioSegment class from [pydub](https://github.com/jiaaro/pydub).

For feature extraction, we'll be using the same NN Google created for their AudioSet, [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish).

Webrtc can't be used without access to a STUN server, so right now Streamlit app is restricted to localhost servers.

### Internal structure
The application structure looks as follows

![Class diagram](images/application_diagram.png)

Streamlit is the entry point for our data source. It has two working modes:
- **"wav"**: saves the audio as a wav file before feature extraction
- **"stream"**: connects directly webrtc AudioSegment to VGGish deep network

## What's next? :crystal_ball:
1. Improve neural network training
2. Move from streamlit app to streamlit api consumer
3. Allow model selection
4. Switch from AudioSet and VGGish to UrbanSounds8k (or similar) and librosa to do our own waveform analysis
5. Mobile app (Flutter?)  
