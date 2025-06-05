
"""
This script defines the Accompany Generator class,
this will be responsible for generating the accompaniment using a trained Transformer model. This class should produce a sequence of musical notes
for multiple instruments (bass,drums,piano,etc.) based on a given input sequence (will later be from user input or a MIDI file)

This class leverages the trained Transformer model's ability to predict subsequent notes in a accompaniment based on the current context.
It should iteratively appendd each predicted note and instrument associated to the existing sequence and feed this back into the model for continued predictions

Currently, this should continue to generate notes until a specified length is reached or a stop condition is met (like a specific note or silence).

there is a tokenizer to encode and decode note sequences to and from the format expected by the Transformer model.

We will be using tensorflow and Keras for the model and training

"""

import tensorflow as tf