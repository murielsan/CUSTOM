# CUSTOM

CUSTOM stands for Classifying Urban Sounds Taken On Madrid. It's a machine learning audio classifying proyect. Basically, we're creating a program that will be able to capture environment sounds and classify them into one of the 10 labels the neural network will be able to detect.

We'll be using Google AudioSet dataset,[AudioSet: An ontology and human-labelled dataset for audio events](https://creativecommons.org/licenses/by-sa/4.0/). Also we'll use their pretrained models, [CNN Architectures for Large-Scale Audio Classification](https://research.google.com/pubs/pub45611.html).

## Packages needed
numpy resampy tensorflow tf_slim six soundfile