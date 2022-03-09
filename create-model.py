import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import os

from numpy.core.records import array
from PIL import Image

training_labels = []

file_path = 'images/'
files = os.listdir(file_path)
image_arr = []

# for file_name in files:
#     image = Image.open(file_path + file_name)
#     # show the image
#     # image.show()
#     price_value = float(file_name.split('~')[0])
#     if price_value < 1:
#         training_labels.append(0)
#     elif price_value >= 1 and price_value < 5:
#         training_labels.append(1)
#     elif price_value >= 5:
#         training_labels.append(2)
#     else:
#         training_labels.append(0)

#     # transform Image into the numpy array
#     image_2_npArray = np.asarray(image)

#     # transform the numpy array into the tensor
#     image_2_tensor = tf.convert_to_tensor(image_2_npArray)
#     image_arr.append(image_2_tensor)
#     break

np_image_arr = np.array(image_arr)
# print(np_image_arr)

np_training_labels = np.array([int(num) for num in training_labels])
np_training_labels = np.array(training_labels)

class_names = ['less than 1 dollar', '1 dollar', '5 dollar']

# build network
model = models.Sequential()
model.add(tf.keras.layers.Conv2D(25, kernel_size=3, strides=1,
          activation='relu', input_shape=(25, 25, 1)))
model.add(tf.keras.layers.PReLU(alpha_initializer='zeros',
          alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(tf.keras.layers.Conv2D(
    25, kernel_size=5, strides=1, activation='relu'))
model.add(tf.keras.layers.PReLU(alpha_initializer='zeros',
          alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(tf.keras.layers.Conv2D(
    25, kernel_size=2, strides=1, activation='relu'))

# add some max pooling here
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=3, strides=1, padding="valid"))

model.add(tf.keras.layers.Conv2D(
    25, kernel_size=1, strides=1, activation='relu'))
model.add(tf.keras.layers.PReLU(alpha_initializer='zeros',
          alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(tf.keras.layers.Conv2D(
    25, kernel_size=1, strides=1, activation='relu'))
model.add(tf.keras.layers.PReLU(alpha_initializer='zeros',
          alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(tf.keras.layers.Conv2D(
    25, kernel_size=1, strides=1, activation='relu'))
model.add(tf.keras.layers.PReLU(alpha_initializer='zeros',
          alpha_regularizer=None, alpha_constraint=None, shared_axes=None))

# print(model.summary())

model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(np_image_arr, np_training_labels, epochs=4)
print('model complete. prediction start!')
# predict_image = model.predict(np_image_arr)
# print(card_labels[0])
# print(len(model.layers))
# print(predict_image)

# predictions = model.predict(np_image_arr)
temp = 1
# print(predictions)
# print(class_names[np.argmax(predictions[temp])])
# """
