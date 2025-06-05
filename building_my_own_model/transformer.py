"""
This script is used to train a model using a Transformer architecture for music accompaniment generation
This relies on the mechanism of attention, differentially weighting the significance of different input elements,

This includes the implementation of the Encoder, Decoder and their respective layers. It also uses a siniusoidal positional encoding function 
so that the model can learn the order of the input sequence

Key Components:
- Transformer: the main model class combining the Encoder and Decoder
- Encoder: [rpcesses the input sequence and generates a context vector
- Decoder: generates the output sequence based on the context vector and previous outputs
EncoderLayer and DecoderLayer: the individual layers of the Encoder and Decoder, implementing multi-head attention and feed-forward networks
_get_angels: a utility function to generate the angles for the positional encoding
- sinusoidal_positional_encoding: a function to generate the positional encoding for the input sequence

]

"""