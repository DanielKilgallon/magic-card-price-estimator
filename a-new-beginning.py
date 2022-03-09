from audioop import bias
import numpy as np
import PIL
import glob
import tensorflow as tf
from torch import channels_last
import math

pngs = glob.glob('all-images/*.png')

ims = {}
for png in pngs:
  ims[png] = np.array(PIL.Image.open(png)) / 255.0

# gets images as a numpy array
questions = np.array([each for each in ims.values()]).astype(np.float32)

# gets the labels as a numpy array
# Windows
solutions = np.array([float(each.split('\\')[-1].split('~')[0]) for each in ims]).astype(np.float32)
# Linux
# solutions = np.array([float(each.split('/')[-1].split('~')[0]) for each in ims]).astype(np.float32)

# Data fixing
solutions = np.log(solutions) / np.log(10)

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(input_shape=(25, 25, 1), filters=32, kernel_size=5, activation='relu', padding="same", kernel_initializer='he_normal'),
  tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu', padding="same", kernel_initializer='he_normal'),
  tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=2, activation='relu', padding="same", kernel_initializer='he_normal'), # Luke's Pooling Layer
  tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding="same", kernel_initializer='he_normal'),
  tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding="same", kernel_initializer='he_normal'),
  tf.keras.layers.Conv2D(filters=128, kernel_size=2, strides=2, activation='relu', padding="same", kernel_initializer='he_normal'), # Luke's Pooling Layer
  tf.keras.layers.Conv2D(filters=128, kernel_size=5, activation='relu', padding="same", kernel_initializer='he_normal'),
  tf.keras.layers.Conv2D(filters=128, kernel_size=5, activation='relu', padding="same", kernel_initializer='he_normal'),
  tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding="same", kernel_initializer='he_normal'),
  tf.keras.layers.Dropout(.50),
  tf.keras.layers.GlobalAveragePooling2D(keepdims=True),
  tf.keras.layers.Conv2D(filters=512, kernel_size=1, activation='relu', padding="same"),
  tf.keras.layers.Dropout(.50),
  tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation=None, padding="same", kernel_initializer='he_normal'),
  tf.keras.layers.Reshape(target_shape=(1,))
])

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer="SGD", metrics=['accuracy'])
model.optimizer.lr = 0.001 # Luke says it's important don't remove for reasons
# ^ but actually important to expose this variable for hyper parameter tuning
history = model.fit(x=questions, y=solutions, epochs=20, batch_size=100, verbose=1)

# Test model

prediction_pngs = glob.glob('prediction-images/*.png')

prediction_ims = {}
for png in prediction_pngs:
  prediction_ims[png] = np.array(PIL.Image.open(png)) / 255.0

prediction_questions = np.array([each for each in prediction_ims.values()]).astype(np.float32)

answers = model.predict(prediction_questions)
# print(prediction_questions)
print("prediction: {}".format(answers[0]))
print("correct: {}".format(np.log(float(prediction_pngs[0].split('\\')[-1].split('~')[0])) / np.log(10)))

# for i in range(len(answers)):
#   print('\ncard: {}'.format(prediction_pngs[i].split('\\')[-1].split('~')[-1].split('.')[0]))
#   print('prediction: ${}'.format(round(float(answers[i][0]), 2)))
#   print('actual: ${}'.format(prediction_pngs[i].split('\\')[-1].split('~')[0]))
# """