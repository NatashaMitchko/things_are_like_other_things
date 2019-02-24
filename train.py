try:
    GUTENBERG = True
    from gutenberg.acquire import load_etext
    from gutenberg.query import get_etexts, get_metadata
    from gutenberg.acquire import get_metadata_cache
    from gutenberg.acquire.text import UnknownDownloadUriException
    from gutenberg.cleanup import strip_headers
    from gutenberg._domain_model.exceptions import CacheAlreadyExistsException
except ImportError:
    GUTENBERG = False
    print("Gutenberg is not installed. See instructions at https://pypi.python.org/pypi/Gutenberg")
from keras.models import Input, Model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
import keras.callbacks
import keras.backend as K
import scipy.misc
import json

import os, sys
import re

from keras.optimizers import RMSprop
import random
import numpy as np
import tensorflow as tf
from keras.utils import get_file

try:
    from io import BytesIO
except ImportError:
    from StringIO import StringIO as BytesIO

# Create the metadata cache
if GUTENBERG:
    cache = get_metadata_cache()
    try:
        print("Building cache")
        cache.populate()
    except CacheAlreadyExistsException:
        pass

# if GUTENBERG:
#     i = 0
#     for text_id in get_etexts("author", "Shakespeare, William"):
#         if i == 10:
#             break
#         print(text_id, list(get_metadata("title", text_id))[0])
#         i += 1
if GUTENBERG:
    shakespeare = strip_headers(load_etext(100))
else:
    path = get_file("shakespeare", "https://storage.googleapis.com/deep-learning-cookbook/100-0.txt")
    shakespeare = open(path).read()

training_text = shakespeare.split("\nTHE END", 1)[-1]
print("Number of entries: ", len(training_text))

# One-hot encoding of characters
chars = list(sorted(set(training_text)))
char_to_index = {ch: idx for idx, ch in enumerate(chars)}
num_chars = len(chars)
print("Unique characters: ", num_chars)

# Create a model that takes sequence of chars and outputs sequence of chars
def char_rnn_model(num_chars, num_layers, num_nodes=512, dropout=0.1):
    input = Input(shape=(None, num_chars), name="input")
    prev = input
    for i in range(num_layers):
        prev = LSTM(num_nodes, return_sequences=True)(prev)
    dense = TimeDistributed(Dense(num_chars, name="dense", activation="softmax"))(prev)
    model = Model(inputs=[input], outputs=[dense])
    optimizer = RMSprop(lr=0.01)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model

model = char_rnn_model(len(chars), num_layers=2, num_nodes=640, dropout=0)
model.summary()

CHUNK_SIZE = 160

def data_generator(all_text, num_chars, batch_size, chunk_size):
    X = np.zeros((batch_size, chunk_size, num_chars))
    Y = np.zeros((batch_size, chunk_size, num_chars))
    while True:
        for row in range(batch_size):
            idx = random.randrange(len(all_text) - chunk_size - 1)
            chunk = np.zeros((chunk_size + 1, num_chars))
            for i in range(chunk_size + 1):
                chunk[i, char_to_index[all_text[idx + i]]] = 1
            X[row, :, :] = chunk[:chunk_size]
            Y[row, :, :] = chunk[1:]
        yield X, Y

# Log the weights
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE_PATH, "logs", "training.log")
csv_logger = keras.callbacks.CSVLogger(LOG, append=False)

early = keras.callbacks.EarlyStopping(monitor="loss",
                                      min_delta=0.03,
                                      patience=3,
                                      verbose=0,
                                      mode="auto")

# Save the weights
MODEL_NAMES = os.path.join(BASE_PATH, "models", ".{epoch:02d}-{val_acc:.2f}.hdf5")
model_checkpoint = keras.callbacks.ModelCheckpoint(MODEL_NAMES, "loss", verbose=1, save_best_only=True)

callbacks = [model_checkpoint, csv_logger, early]

# Train the model so that each character has a chance of being seen
BATCH_SIZE = 256
model.fit_generator(
    data_generator(training_text, len(char_to_index), batch_size=BATCH_SIZE, chunk_size=CHUNK_SIZE),
    epochs=40,
    callbacks=callbacks,
    steps_per_epoch= 2 * len(training_text) / (BATCH_SIZE * CHUNK_SIZE),
    verbose=2
)
