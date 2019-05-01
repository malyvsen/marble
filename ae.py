#%% imports
from tqdm import trange
import tensorflow as tf
import images
import utils


#%% aechitecture
class Encoder:
    with tf.variable_scope('encoder'):
        inputs = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))

        layers = [inputs]
        layers.append(tf.layers.conv2d(layers[-1], filters=32, kernel_size=5, padding='same', activation=tf.nn.relu))
        layers.append(tf.layers.max_pooling2d(layers[-1], pool_size=(2, 2), strides=(2, 2), padding='same'))
        layers.append(tf.layers.conv2d(layers[-1], filters=32, kernel_size=3, padding='same', activation=tf.nn.relu))
        layers.append(tf.layers.max_pooling2d(layers[-1], pool_size=(2, 2), strides=(2, 2), padding='same'))
        layers.append(tf.layers.conv2d(layers[-1], filters=16, kernel_size=3, padding='same', activation=tf.nn.relu))
        layers.append(tf.layers.max_pooling2d(layers[-1], pool_size=(2, 2), strides=(2, 2), padding='same'))
        layers.append(tf.layers.conv2d(layers[-1], filters=16, kernel_size=3, padding='same', activation=tf.nn.relu))
        layers.append(tf.layers.max_pooling2d(layers[-1], pool_size=(2, 2), strides=(2, 2), padding='same'))

        outputs = layers[-1]


class Decoder:
    with tf.variable_scope('decoder'):
        inputs = Encoder.outputs

        layers = [inputs]
        layers.append(utils.zoom_nearest_neighbor(layers[-1], zoom=2))
        layers.append(tf.layers.conv2d(layers[-1], filters=16, kernel_size=3, padding='same', activation=tf.nn.relu))
        layers.append(utils.zoom_nearest_neighbor(layers[-1], zoom=2))
        layers.append(tf.layers.conv2d(layers[-1], filters=16, kernel_size=3, padding='same', activation=tf.nn.relu))
        layers.append(utils.zoom_nearest_neighbor(layers[-1], zoom=2))
        layers.append(tf.layers.conv2d(layers[-1], filters=32, kernel_size=3, padding='same', activation=tf.nn.relu))
        layers.append(utils.zoom_nearest_neighbor(layers[-1], zoom=2))
        layers.append(tf.layers.conv2d(layers[-1], filters=32, kernel_size=5, padding='same', activation=tf.nn.relu))

        logits = tf.layers.conv2d(layers[-1], filters=3, kernel_size=5, padding='same')
        layers.append(logits)
        outputs = tf.nn.sigmoid(logits) * 255
        layers.append(outputs)

        targets = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))


mse_loss = tf.reduce_mean(tf.square(Decoder.outputs - Decoder.targets)) / (255 ** 2)
ssim_loss = -tf.reduce_mean(tf.image.ssim(Decoder.outputs, Decoder.targets, max_val=255)) / 2
loss_weight = tf.placeholder(dtype=tf.float32, shape=(2,))
loss = mse_loss * loss_weight[0] + ssim_loss * loss_weight[1]
optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)


#%% training/loading
import os
session = tf.Session()
saver = tf.train.Saver()
save_location  = 'checkpoints/ae.ckpt'

def train(num_steps, demo_interval=16, batch_size=16, loss_transition_steps=64):
    for i in trange(num_steps):
        if i % demo_interval == 0:
            decoded = session.run(Decoder.outputs, feed_dict={Encoder.inputs: images.batch(4)[0]})
            images.show(decoded)
        inputs, targets = images.batch(batch_size)
        feed = {Encoder.inputs: inputs, Decoder.targets: targets, loss_weight: utils.loss_weight(i, loss_transition_steps)}
        session.run(optimizer, feed_dict=feed)

if os.path.isfile(save_location + '.index'):
    saver.restore(session, save_location)
else:
    session.run(tf.global_variables_initializer())
    train(num_steps=1024)
    saver.save(session, save_location)


#%% test
import skimage
from glob import glob


def test(path, save=False):
    for file in glob(path):
        if not os.path.isfile(file):
            continue
        original = images.to_rgb(skimage.io.imread(file))
        decoded = session.run(Decoder.outputs, feed_dict={Encoder.inputs: [original]})
        images.show(decoded, save_as=file.replace('.', '_marbled.') if save else None)



test('test/*', save=True)
