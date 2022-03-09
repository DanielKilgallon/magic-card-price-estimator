import requests
import math
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# card_obj_fields = ['cmc', 'color_identity', 'type_line', 'power','toughness', 'oracle_text']
card_obj_fields = ['oracle_text']
base_url = 'https://api.scryfall.com/cards/oracle/'

image_size = [25, 25]

image_arr = []

PRICE_CUTOFF = float(1.0)

download_url = requests.get('https://api.scryfall.com/bulk-data/27bf3214-1271-490b-bdfe-c0be6c23d02e').json()['download_uri']
json_data = requests.get(download_url).json()

for card_obj in json_data:
    price_value = card_obj['prices']['usd']
    # skip commander illegal cards, reserved list cards, and blank or 0 prices
    if (card_obj['legalities']['commander'] == 'not_legal' or
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
    price_value = float(price_value)
    folder_prefix = None
    if price_value < PRICE_CUTOFF:
        folder_prefix="LessThanCutOff"
    elif price_value >= PRICE_CUTOFF:
        folder_prefix="GreaterThanCutOff"
    else:
        folder_prefix="LessThanCutOff"

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
    # Converting numbers to uint8 for image 
    resized_image_data = np.array(resized, dtype=np.uint8)
    img = Image.fromarray(resized_image_data)

    # Saving Image
    if not folder_prefix is None: 
        os.makedirs(f'training-images/{folder_prefix}/', exist_ok=True)
        img.save(f'training-images/{folder_prefix}/{image_name}.png')
    else:
        img.save(f'training-images/{image_name}.png')