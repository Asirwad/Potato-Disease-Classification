import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Resizing, Rescaling
from tensorflow.keras import layers

BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS = 3

num_of_class = 3

resize_and_rescale_layer = Sequential([
    Resizing(IMAGE_SIZE, IMAGE_SIZE),
    Rescaling(1. / 255),
])


def get_model_arch(input_shape):
    model = Sequential([
        resize_and_rescale_layer,
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_of_class, activation='softmax'),
    ])
    return model
