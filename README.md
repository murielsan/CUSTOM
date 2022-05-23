# CUSTOM

CUSTOM stands for Classifying Urban Sounds Taken On Madrid. It's a machine learning audio classifying proyect. Basically, I'm building a program that will be able to capture environmental sounds and classify them into one of the 10 labels the neural network will be able to detect.

I'll be using Google AudioSet dataset,[AudioSet: An ontology and human-labelled dataset for audio events](https://creativecommons.org/licenses/by-sa/4.0/) for training the different models. Also we'll use their pretrained models, [CNN Architectures for Large-Scale Audio Classification](https://research.google.com/pubs/pub45611.html).

## Data preparation
Google's Audioset contains thousands of 10s audios (extracted from youtube videos) and 527 different classes provided as tensorflow records. After loading the balanced set records into variables, After analyzing the data, these are the classes with the most number of samples:

![top 30 classes](images/top30classes.png)

### Classes selected
I've selected the following 10 classes to be identified by our model:

- Vehicle horn, car horn, honking
- Children playing
- Dog
- Jackhammer
- Siren
- Traffic noise, roadway noise
- Subway, metro, underground
- Walk, footsteps
- Chatter
- Bird

These are common sounds you could hear when you walk along Madrid. We'll be training our models to identify these sounds

### Multiclass classification


## Packages needed
numpy resampy tensorflow tf_slim six soundfile