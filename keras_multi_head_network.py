import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from util.paths import ensure_dir

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV3Small

import wandb
from wandb.keras import WandbCallback

X = np.random.randint(1, 256, size=64)

ydiv2 = (X % 2 == 0).astype(int)
ydiv3 = (X % 3 == 0).astype(int)
ydiv5 = (X % 5 == 0).astype(int)
ydiv7 = (X % 7 == 0).astype(int)

Y1 = np.transpose(np.vstack([ydiv2, ydiv3, ydiv5, ydiv7]))
Y2 = X

wandb.init(project='keras_multi_head_test', entity='lehl')

inp = keras.Input(shape=(1,))
x = layers.Dense(16, activation='sigmoid')(inp)
x = layers.Dense(16, activation='sigmoid')(x)
x = layers.Dense(16, activation='sigmoid')(x)
o1 = layers.Dense(4, activation='sigmoid', name='classification_out')(x)
o2 = layers.Dense(1, activation='relu', name='regression_out')(x)

model = keras.Model(inputs=inp, outputs=[o1, o2])

model.summary()
model.compile(
    optimizer='adam',
    loss={
        'classification_out': keras.losses.CategoricalCrossentropy(from_logits=True),
        'regression_out': keras.losses.MeanSquaredError()
    },
    metrics={
        'classification_out': ['accuracy'],
        'regression_out': keras.metrics.MeanSquaredError()
    }
)

model.fit(X, {'classification_out': Y1, 'regression_out': Y2}, epochs=100, verbose=1, callbacks=[WandbCallback()])


print(model.predict([2, 3, 5, 7, 128, 256]))
# import code; code.interact(local=dict(globals(), **locals()))


