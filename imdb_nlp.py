import numpy as np
import scipy
import json
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, LSTM, Bidirectional, GlobalAveragePooling1D, \
    GlobalAveragePooling2D, Conv1D, Flatten, RNN, GRU, Embedding
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

########################
# Sentence preprocessing
########################

# Let's first create global variables
vocab_size = 10000
oov_token = ''
padding_style = 'post'
trunc_style = 'post'
max_length = 120
embedding_dim = 16

# let's initialize the tokenizer class

tokenizer = Tokenizer(oov_token=oov_token, num_words=vocab_size)
tokenizer.fit_on_texts(train_sentence)
word_index = tokenizer.word_index

# turn sentence into sequences
train_sequence = tokenizer.texts_to_sequences(train_sentence)
print(len(train_sequence[3]))

# let's pad the sequences
train_padded = pad_sequences(train_sequence, padding=padding_style,
                             truncating=trunc_style,
                             maxlen=max_length)

test_sequence = tokenizer.texts_to_sequences(test_sentence)
test_padded = pad_sequences(test_sequence, padding=padding_style,
                            truncating=trunc_style,
                            maxlen=max_length)
################
# Model Creation
################

# let's build a simple model

model = keras.Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    #GlobalAveragePooling1D(),
    Bidirectional(LSTM(64)),
    Flatten(),
    Dense(32, activation="relu"),
    Dropout(.2),
    Dense(1, activation="sigmoid")
])

model.summary()

model.compile(loss="binary_crossentropy",
              optimizer="Adam",
              metrics=["accuracy"])

num_epochs = 10

history = model.fit(train_padded, train_label, epochs=num_epochs,
                    validation_data=(test_padded, test_label), verbose=2)
