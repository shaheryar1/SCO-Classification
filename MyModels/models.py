from keras import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D,AveragePooling2D,Flatten,Dropout,Conv2D
import os
import keras
from keras.applications import MobileNetV2,ResNet50
from config import CLASSES
import cv2
import numpy as np
# from Utils.Visualize import plotConfusionMatrix,plot_images
import matplotlib.pyplot as plt

INPUT_SIZE=(150,150)

def mobilenet_v2_custom(NUM_CLASSES,input_shape =(150, 150, 3)):
    base_model = MobileNetV2(include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(NUM_CLASSES, activation='softmax')(x)
#     preds = Flatten()(x)
    # Create model
    model = Model(inputs=base_model.input, outputs=preds)
    return model

def Resnet50_custom(NUM_CLASSES,input_shape =(150, 150, 3)):
    base_model = ResNet50(include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(NUM_CLASSES, activation='softmax')(x)

    # Create model
    model = Model(inputs=base_model.input, outputs=preds)
    return model

def inference_np(model,img):
    img = img.copy()
    img = cv2.resize(img, (150, 150))
    img_array = preprocess_input(np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), axis=0))
    p = model.predict(img_array)

    return p

def inferenceImage(model,path):
    img = load_image(img_path=path,expand_dim=True)
    p=model.predict(img)
    predicted_class=CLASSES[np.argmax(p)]
    return predicted_class

def load_image(img_path,expand_dim=False):
    img = image.load_img(img_path, target_size=INPUT_SIZE)
    img_array = image.img_to_array(img)
    if(expand_dim):
        img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)