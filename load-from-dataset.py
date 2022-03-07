import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


train_ds = tf.keras.utils.image_dataset_from_directory(
    "training-images/",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(25, 25),
    batch_size=100,
    color_mode="grayscale"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "training-images/",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(25, 25),
    batch_size=100,
    color_mode="grayscale"
)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

# Defines the model
model = Sequential([
    layers.Conv2D(16, 1,padding='same', activation='relu'),
    layers.AveragePooling2D(),
    layers.Conv2D(32, 1, padding='same', activation='relu'),
    layers.AveragePooling2D(),
    layers.Conv2D(64, 1, padding='same', activation='relu'),
    layers.AveragePooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# Generates the model object
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Trains model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# SHOWS ACCURACY AND LOSS GRAPHS
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

prediction_images = os.listdir("prediction-images/")

for image_path in prediction_images:
    img = tf.keras.utils.load_img(
        "prediction-images/" + image_path, target_size=(25, 25), color_mode="grayscale"
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    price_name_tuple = image_path.split("~")
    print("price: {}".format(price_name_tuple[0]))
    print("card: {}".format(price_name_tuple[1]))
    print("score: {}".format(class_names[np.argmax(score)]))
    print("confidence: {}\n".format(100 * np.max(score)))