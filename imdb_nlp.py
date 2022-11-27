import numpy as np
import scipy
import json
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, LSTM, Bidirectional, GlobalAveragePooling1D, \
    GlobalAveragePooling2D, Conv1D, Flatten, RNN, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

imdb, info = tfds.load("imdb_reviews",with_info=True,
                       as_supervised=True)
print(info)