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

imdb, info = tfds.load("imdb_reviews", with_info=True,
                       as_supervised=True)
print(info)
for example in imdb['train'].take(3):
    print(example)
# get the train and test sets
train_data, test_data = imdb['train'], imdb['test']

# split the data into features and labels for train and test
train_sentence = []
test_sentence = []

train_label = []
test_label = []

for k, l in train_data:
    train_sentence.append(k.numpy().decode('utf8'))
    train_label.append(l.numpy())

for k, l in test_data:
    test_sentence.append(k.numpy().decode('utf8'))
    test_label.append(l.numpy())

test_label = np.array(test_label)
train_label = np.array(train_label)

