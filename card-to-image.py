import math
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import requests

from PIL import Image
from numpy.core.records import array

# card_obj_fields = ['cmc', 'color_identity', 'type_line', 'power','toughness', 'oracle_text']
card_obj_fields = ['oracle_text']
base_url = 'https://api.scryfall.com/cards/oracle/'

image_size = [25, 25]

card_counter = 0
card_stop_spot = 10

image_arr = []
training_labels = []
card_labels = []

download_url = requests.get('https://api.scryfall.com/bulk-data/27bf3214-1271-490b-bdfe-c0be6c23d02e').json()['download_uri']
json_data = requests.get(download_url).json()

for card_obj in json_data:
    price_value = card_obj['prices']['usd']
    # skip standard legal cards, commander illegal cards, reserved list cards, and blank or 0 prices
    if (card_obj['legalities']['standard'] == 'legal' or
        card_obj['legalities']['commander'] == 'not_legal' or
        not price_value or (price_value and float(price_value) == 0.0) or
        card_obj['reserved'] == True):
        continue

    # print(f'creating image for {card_obj["name"]}')
    card_data_arr = []
    image_data = []

    image_name = str(price_value) + '~' + ((((str(card_obj["name"]).replace('?','')).replace('//','')).replace('"','')).replace(':',''))
    invalid_oracle_text = False
    # gets all fields from card json, or sets as zero
    for field in card_obj_fields:
        if field in card_obj.keys():
            data = card_obj[field]
            if isinstance (data, float):
                data = round(data)
            if isinstance (data, list):
                data = ''.join(data)
            if not data:
                invalid_oracle_text = True
                continue
            card_data_arr += [ord(c) % 256 for c in str(data)]
        else:
            # if the card has multiple faces
            if len(card_obj['card_faces']) > 1:
                data = card_obj['card_faces'][0][field]
                data += card_obj['card_faces'][1][field]
                card_data_arr += [ord(c) % 256 for c in str(data)]
                # print(f'{card_obj["name"]} is double faced!')
            else:
                card_data_arr += [0]
                invalid_oracle_text = True
                # print(f'{field} does not exist on card {card_obj["name"]}, skipping...')
    if invalid_oracle_text == True:
        continue
    card_labels.append(card_obj["name"])
    price_value = float(price_value)
    if price_value < 1:
        training_labels.append(0)
    elif price_value >= 1 and price_value < 5:
        training_labels.append(1)
    elif price_value >= 5:
        training_labels.append(2)
    else:
        training_labels.append(0)

    array_size = math.ceil(math.sqrt(len(card_data_arr)))
    card_spot = 1
    image_row = []
    complete_row = False
    # converts 1d array into closest 2d array of sqrt size of 1d length
    for ascii_val in card_data_arr:
        if card_spot == array_size:
            card_spot = 1
            image_data.append(image_row)
            image_row = []
            complete_row = True
        if card_spot < array_size:
            card_spot += 1
            image_row.append(ascii_val)
            complete_row = False
    # if ending array is ragged, fix
    if complete_row == False:
        image_fix_len = len(image_row)
        for i in range(image_fix_len, array_size - 1):
            image_row.append(0)
        image_data.append(image_row)

    # Tensorflow Resize
    tf_arr = tf.constant(image_data)
    tf_arr = tf_arr[tf.newaxis, ..., tf.newaxis]
    tf_arr.shape.as_list()  # [batch, height, width, channels]
    resized = tf.image.resize(tf_arr, image_size)[0,...,0].numpy()
    resized_image_data = np.array(resized)
    image_arr.append(resized_image_data)

np_image_arr = np.array(image_arr)

np_training_labels = np.array([int(num) for num in training_labels])
np_training_labels = np.array(training_labels)

class_names = ['less than 1 dollar', '1 dollar', '5 dollar']

# build network
model = models.Sequential()
model.add(tf.keras.layers.Conv2D(25, kernel_size=3, strides=1, activation='relu', input_shape=(25, 25, 1)))
model.add(tf.keras.layers.PReLU(alpha_initializer='zeros',alpha_regularizer=None,alpha_constraint=None,shared_axes=None))
model.add(tf.keras.layers.Conv2D(25, kernel_size=5, strides=1, activation='relu'))
model.add(tf.keras.layers.PReLU(alpha_initializer='zeros',alpha_regularizer=None,alpha_constraint=None,shared_axes=None))
model.add(tf.keras.layers.Conv2D(25, kernel_size=2, strides=1, activation='relu'))

# add some max pooling here
model.add(tf.keras.layers.MaxPooling2D(pool_size=3, strides=1, padding="valid"))

model.add(tf.keras.layers.Conv2D(25, kernel_size=1, strides=1, activation='relu'))
model.add(tf.keras.layers.PReLU(alpha_initializer='zeros',alpha_regularizer=None,alpha_constraint=None,shared_axes=None))
model.add(tf.keras.layers.Conv2D(25, kernel_size=1, strides=1, activation='relu'))
model.add(tf.keras.layers.PReLU(alpha_initializer='zeros',alpha_regularizer=None,alpha_constraint=None,shared_axes=None))
model.add(tf.keras.layers.Conv2D(25, kernel_size=1, strides=1, activation='relu'))
model.add(tf.keras.layers.PReLU(alpha_initializer='zeros',alpha_regularizer=None,alpha_constraint=None,shared_axes=None))

print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# history = model.fit(np_image_arr, np_training_labels, epochs=4)
print('model complete. prediction start!')
# predict_image = model.predict(np_image_arr)
# print(card_labels[0])
print(len(model.layers))
# print(predict_image)

"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(25, 25)),  # input layer
    keras.layers.Dense(128, activation='prelu'), # hidden layer
    keras.layers.Dense(3, activation='softmax'), # output layer
])

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(np_image_arr, np_training_labels, epochs=4)

print('model complete. prediction start!')
predictions = model.predict(np_image_arr)
temp = 1
print(card_labels[temp])
print(class_names[np.argmax(predictions[temp])])
    # Converting numbers to uint8 for image 
    # resized_image_data = np.array(resized, dtype=np.uint8)
    # img = Image.fromarray(resized_image_data)
    # img.save(f'images/{image_name}.png')
#"""