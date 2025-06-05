
"""
This script defines the accompany preprocessor class, this class is responsible for preprocessing the input data for the accompaniment generation model.
It will handle the conversion of MIDI files into a format suitable for training the model, including extracting note sequences, instrument information, and any necessary transformations.
This will also use the REMI style format for representing musical notes and instruments, which is a common format for music generation tasks.

I will also make sure when handling the MIDI files, to make sure to safely handle the different instruments and their respective notes for the chosen instruments
since last time I tried to use a MIDI file with multiple instruments and split them into piano,drums and bass it was messy and there were overlapping notes which made the model messy

This will handle the entire preprocessing pipeline including loading the dataset, extracting relevant features, toeknizing and encoding the data
and forming input-target pairs for training the model. It will als will handle any necessary data augmentation or transformation to improve the model's performance. 
As well as handling padding sequences to a ensure consistent input shape for the model (check if this is needed)

Key Features:
- Load MIDI files and extract note sequences for multiple instruments (bass, drums, piano, etc.)
- Convert note sequences into REMI style format
- Tokenize and encode the note sequences for input to the Transformer model
- Handle padding and sequence length management
- Prepare input-target pairs for training the model



"""

import json 
import numpy as np 
import tensorflow as tf
from keras.preprocessing.text import Tokenizer

