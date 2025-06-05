"""
This file is responsible for training the Transformer model for music accompaniment generation. It will include
functions to calculate loss, perform training steps and manage the training process over multiple epochs.

The training process uses a custom implementation of the Transformer model, defined in the `transformer.py` file. and prepares the data using the `accompanypreprocessor.py` script.

Global vars like # of epochs, batch size, and path to the dataset will be defined here.

Currently the training will be done on a single GPU, but it can be extended to multiple GPUs if needed.

This does not deal with masking of padded vals in the encoder/decoder, and we need to look at the look ahead masking for the decoder to prevent it from seeing future tokens during training.

Key Features:
- calculate_loss_function: Computes the loss for the model based on the predictions and true values.
- train_step: Performs a single training step, including forward pass, loss calculation, and backpropagation
- train_model: Manages the training loop, iterating over epochs and batches, and calling the train_step function.
- _right_pad_sequences: Pads sequences to the right to ensure consistent input shape for the model.`

This script will instantiatiate the Transformer model, load the preprocessed data from the `accompanypreprocessor.py` script,
and train the model using the defined training loop. It will also handle saving the trained model and any necessary checkpoints.

and finally it will generate a sample accompaniment using the trained model to verify that the training was successful and the model can generate music.

"""

import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from accompanygenerator import AccompanyGenerator
from accompanypreprocessor import AccompanyPreprocessor

# Gloabal vars
EPOCHS = 50
BATCH_SIZE = 32
DATASET_PATH = 'path/to/your/dataset'  # Update this to your dataset path
MAX_POSITIONS_IN_POSITIONAL_ENCODING = 512  # Maximum number of positions in positional encoding

# Loss function and optimizer, this will be using ADAM optimizer and sparse categorical crossentropy since we are dealing with a sequence of tokens
sparse_categorical_crossentropy = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
optimizer = Adam(learning_rate=0.001)

