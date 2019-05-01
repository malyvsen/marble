import tensorflow as tf
import numpy as np


def zoom_nearest_neighbor(tensor, zoom, **kwargs):
    return tf.image.resize_nearest_neighbor(tensor, size=tf.shape(tensor)[1:3] * zoom, **kwargs)


# used to weigh between MSE loss (which is good for early training) and SSIM loss (which works better later on)
def loss_weight(step, saturation_steps):
    weight = np.interp(step, [0, saturation_steps], [0, 1])
    return np.array([1 - weight, weight])
