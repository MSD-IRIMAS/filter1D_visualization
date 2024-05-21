import tensorflow as tf
import numpy as np
import random
import os


def znormalisation(x):

    stds = np.std(x, axis=1, keepdims=True)
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=1, keepdims=True)) / stds
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True))


def create_directory(directory_path):

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def generate_distinct_colors(num_colors):
    colors = []
    for _ in range(num_colors):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        
        hex_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        
        colors.append(hex_color)
    
    return colors

accepted_layers = (
    tf.keras.layers.Conv1D,
    tf.keras.layers.SeparableConv1D,
    tf.keras.layers.DepthwiseConv1D,
    tf.keras.layers.Conv1DTranspose,
)
