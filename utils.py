import tensorflow as tf
import numpy as np


def zoom_nearest_neighbor(tensor, zoom, **kwargs):
    return tf.image.resize_nearest_neighbor(tensor, size=tf.shape(tensor)[1:3] * zoom, **kwargs)


def max_pooling(tensor, pool_size, padding='same', **kwargs):
    return tf.layers.max_pooling2d(tensor, pool_size=(pool_size, pool_size), strides=(pool_size, pool_size), padding=padding, **kwargs)


def average(values, weights):
    return tf.multiply(values, weights) / tf.reduce_sum(weights)
