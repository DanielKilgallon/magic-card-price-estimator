import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import glob
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

pngs = glob.glob('data/training-images/*.png')

ims = {}
for png in pngs:
    ims[png]=np.array(PIL.Image.open(png))

questions = np.array([each for each in ims.values()]).astype(np.float32)
solutions = np.array([float(each.split('/')[2].split('~')[0]) for each in ims]).astype(np.float32)

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(input_shape=(25, 25, 1), filters=3, kernel_size=3, activation='relu'),
  tf.keras.layers.Conv2D(filters=3, kernel_size=15, activation='relu'),
  tf.keras.layers.AveragePooling2D(),
  tf.keras.layers.Conv2D(filters=3, kernel_size=15, activation='relu'),
  tf.keras.layers.Conv2D(filters=3, kernel_size=15, activation='relu'),
  tf.keras.layers.AveragePooling2D(),
  tf.keras.layers.Conv2D(filters=3, kernel_size=15, activation='relu'),
  tf.keras.layers.Conv2D(filters=3, kernel_size=15, activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(units=512, activation='relu'),
  tf.keras.layers.Dense(units=256, activation='relu'),
  tf.keras.layers.Dense(units=64, activation='relu'),
  tf.keras.layers.Dense(units=1)
])


model.compile(loss='mean_squared_error', optimizer="adam")

# model.compile(
#     optimizer='adam',
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=['accuracy']
# )

history = model.fit(questions, solutions, epochs=30, batch_size=200, verbose=1)
"""
test_answers = model.predict(test_questions)



print(pngs[0])
print(questions[0])
print(solutions[0])
# """