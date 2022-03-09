import numpy as np
import PIL
import glob
import tensorflow as tf

pngs = glob.glob('all-images/*.png')

ims = {}
for png in pngs:
  ims[png] = np.array(PIL.Image.open(png))

# gets images as a numpy array
questions = np.array([each for each in ims.values()]).astype(np.float32)

# gets the labels as a numpy array
# Windows
solutions = np.array([float(each.split('\\')[-1].split('.')[0]) for each in ims]).astype(np.float32)
# Linux
# solutions = np.array([float(each.split('/')[-1].split('~')[0]) for each in ims]).astype(np.float32)

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(input_shape=(25, 25, 1), filters=4, kernel_size=2, activation='relu'),
  tf.keras.layers.Conv2D(filters=4, kernel_size=2, activation='relu'),
  tf.keras.layers.AveragePooling2D(),
  tf.keras.layers.Conv2D(filters=4, kernel_size=2, activation='relu'),
  tf.keras.layers.Conv2D(filters=4, kernel_size=2, activation='relu'),
  tf.keras.layers.AveragePooling2D(),
  tf.keras.layers.Conv2D(filters=4, kernel_size=2, activation='relu'),
  tf.keras.layers.Conv2D(filters=4, kernel_size=2, activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(units=512, activation='relu'),
  tf.keras.layers.Dense(units=256, activation='relu'),
  tf.keras.layers.Dense(units=64, activation='relu'),
  tf.keras.layers.Dense(units=1)
])

model.compile(loss='mean_squared_error', optimizer="adam", metrics=['accuracy'])

history = model.fit(questions, solutions, epochs=3, batch_size=100, verbose=1)

# Test model

prediction_pngs = glob.glob('prediction-images/*.png')

prediction_ims = {}
for png in prediction_pngs:
  prediction_ims[png] = np.array(PIL.Image.open(png))

prediction_questions = np.array([each for each in prediction_ims.values()]).astype(np.float32)

answers = model.predict(prediction_questions)

for i in range(len(answers)):
  print('\ncard: {}'.format(prediction_pngs[i].split('\\')[-1].split('~')[-1].split('.')[0]))
  print('prediction: ${}'.format(round(float(answers[i][0]), 2)))
  print('actual: ${}'.format(prediction_pngs[i].split('\\')[-1].split('~')[0]))
