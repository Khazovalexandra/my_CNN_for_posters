from PIL import Image, ImageFilter
import os, sys
import pandas as pd
import numpy as np
import pathlib
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import plot_model
from keras.datasets import mnist
import matplotlib.pylab as plt


data_dir = pathlib.Path("./movies_posters/")

genres_set = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])

#def augment(image, seed):
#    image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=seed)
#    image = tf.image.stateless_random_crop(image, size=[38, 56, 3], seed=seed)
#    image = tf.image.central_crop(image, central_fraction=0.5)

for g in genres_set:
    l = list(data_dir.glob(g+'/*.jpg'))
    count = len(l)
    if (count <= 5001):
        print(g)
        print(count)
        k = len(l)
        while (count <= 5000):
            image = Image.open(l[random.randint(0, k-1)])
            print(image)
            new_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            new_image = new_image.crop((5, 5, 30, 50))
            new_image = new_image.resize((38, 56), Image.ANTIALIAS)
            #new_image = new_image.rotate(7)
            new_image = new_image.filter(ImageFilter.SHARPEN)
            new_image.save('movies_posters/' + g + '/' + str(count+1)+ '.jpg')
            count+=1