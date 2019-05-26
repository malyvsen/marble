import tensorflow as tf
import numpy as np


def zoom_nearest_neighbor(tensor, zoom, **kwargs):
    return tf.image.resize_nearest_neighbor(tensor, size=tf.shape(tensor)[1:3] * zoom, **kwargs)


def pool(tensor, pool_type, pool_size, padding='SAME', **kwargs):
    return tf.nn.pool(tensor, window_shape=(pool_size, pool_size), pooling_type=pool_type, strides=(pool_size, pool_size), padding=padding, **kwargs)


def average(values, weights):
    return tf.multiply(values, weights) / tf.reduce_sum(weights)
