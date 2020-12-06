
import os
import sys
import cv2
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.models import model_from_json
from PIL import Image
import numpy as np
import matplotlib.pylab as plt

def image_preprocessing(path, shape=(224, 224), data_format='channels_last'):
    image = Image.open(path)
    image = image.resize(shape)
    image = np.asarray(image, dtype=np.float32)
    image /= 225
    image = np.expand_dims(image, axis=0)
    return image

def show_layer_out(layer_name, activations):
    images_per_row = 10
    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()

json_file = open("./model.json")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./model.h5")
print("Loaded model from disk")

#######Image array#########
original_image = image_preprocessing('./Jellyfish.jpg')
#######activationa and layer names #####
layer_outputs = [layer.output for layer in loaded_model.layers[2:15]]
activation_model = Model(inputs=loaded_model.input,outputs=layer_outputs)
print(activation_model.summary())
activations = activation_model.predict(original_image)
layer_names = []
for layer in loaded_model.layers[2:15]:
    print(layer.name)
    layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot
layer_names = layer_names[:5]
show_layer_out(layer_names, activations)
