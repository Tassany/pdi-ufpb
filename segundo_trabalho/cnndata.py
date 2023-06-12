import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import glob
import random
import pathlib

from PIL import Image


import tensorflow as keras
from keras import layers
from keras.models import Sequential

batch_size = 32
img_height = 640
img_width = 480

dataset_url_y = "https://drive.google.com/u/0/uc?id=1Y8uGtvvlfb-Qzaoq6GjrKVW3-wQR3f_v&export=download"
dataset_url_z = "https://drive.google.com/u/0/uc?id=1mCUQ9_my5UwKQMOXwz0S-QMXgoMTcSC0&export=download"
dataset_url_all = "https://drive.google.com/u/0/uc?id=10yGp5PkZAjjcVYO0RDHM1fOQwt8OUtCR&export=download"

dataset_url_x = "https://drive.google.com/u/0/uc?id=1kTd6gORuUlvsPzjZ8MumayDJXW8_rZx0&export=download"
data_dir = tf.keras.utils.get_file('eixo_x', origin=dataset_url_x, untar=True)
data_dir = pathlib.Path(data_dir)

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


class_names = train_ds.class_names
print(class_names)

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)


normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)


data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])


epochs = 30
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# eixoy_url = "https://raw.githubusercontent.com/Tassany/pdi-ufpb/main/segundo_trabalho/eixo_y/accel_80_F6.csv_eixo_y/grafico_intervalo_10_Y.png"
# eixoy_path = tf.keras.utils.get_file('eixo_y', origin=eixoy_url, untar=True)
eixoy_path = "/home/tassany/Desktop/UFPB/PDI/pdi-ufpb/segundo_trabalho/eixo_x/accel_80_F0_eixo_x/grafico_intervalo_11_x.png"
# img = tf.keras.utils.load_img(
#     eixoy_path, target_size=(img_height, img_width)
# )
# img_array = tf.keras.utils.img_to_array(img)
img = Image.open(eixoy_path)
img = img.convert("RGB")
img = img.resize((img_width, img_height))

# Converta a imagem em um tensor do TensorFlow
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
