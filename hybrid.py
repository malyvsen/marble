#%% imports
from tqdm import trange
import tensorflow as tf
import numpy as np
import random
import images
import utils


#%% architecture
class Generator:
    with tf.variable_scope('encoder'):
        inputs = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))

        representations = [inputs]
        representations.append(tf.layers.conv2d(representations[-1], filters=32, kernel_size=5, padding='same', activation=tf.nn.relu))
        representations.append(tf.layers.max_pooling2d(representations[-1], pool_size=(2, 2), strides=(2, 2), padding='same'))
        representations.append(tf.layers.conv2d(representations[-1], filters=32, kernel_size=3, padding='same', activation=tf.nn.relu))
        representations.append(tf.layers.max_pooling2d(representations[-1], pool_size=(2, 2), strides=(2, 2), padding='same'))
        representations.append(tf.layers.conv2d(representations[-1], filters=16, kernel_size=3, padding='same', activation=tf.nn.relu))
        representations.append(tf.layers.max_pooling2d(representations[-1], pool_size=(2, 2), strides=(2, 2), padding='same'))
        representations.append(tf.layers.conv2d(representations[-1], filters=16, kernel_size=3, padding='same', activation=tf.nn.relu))
        representations.append(tf.layers.max_pooling2d(representations[-1], pool_size=(2, 2), strides=(2, 2), padding='same'))

        encoding = representations[-1]

    with tf.variable_scope('decoder'):
        representations.append(utils.zoom_nearest_neighbor(representations[-1], zoom=2))
        representations.append(tf.layers.conv2d(representations[-1], filters=16, kernel_size=3, padding='same', activation=tf.nn.relu))
        representations.append(utils.zoom_nearest_neighbor(representations[-1], zoom=2))
        representations.append(tf.layers.conv2d(representations[-1], filters=16, kernel_size=3, padding='same', activation=tf.nn.relu))
        representations.append(utils.zoom_nearest_neighbor(representations[-1], zoom=2))
        representations.append(tf.layers.conv2d(representations[-1], filters=32, kernel_size=3, padding='same', activation=tf.nn.relu))
        representations.append(utils.zoom_nearest_neighbor(representations[-1], zoom=2))
        representations.append(tf.layers.conv2d(representations[-1], filters=32, kernel_size=5, padding='same', activation=tf.nn.relu))

        logits = tf.layers.conv2d(representations[-1], filters=3, kernel_size=5, padding='same')
        representations.append(logits)
        outputs = tf.nn.sigmoid(logits) * 255
        representations.append(outputs)

    vars = tf.trainable_variables(scope='encoder') + tf.trainable_variables(scope='decoder')

    targets = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))
    mse_loss = tf.reduce_mean(tf.square(outputs - targets)) / (255 ** 2) # normalized into interval of size 1
    # training definition continued after discriminator is initialized


class Discriminator:
    with tf.variable_scope('discriminator'):
        inputs = Generator.outputs # will be fed directly if it's a real image

        representations = [inputs]
        wide = tf.layers.conv2d(representations[-1], filters=8, kernel_size=(15, 5), padding='same', activation=tf.nn.relu)
        high = tf.layers.conv2d(representations[-1], filters=8, kernel_size=(5, 15), padding='same', activation=tf.nn.relu)
        representations.append(tf.concat((wide, high), axis=-1))
        representations.append(tf.layers.max_pooling2d(representations[-1], pool_size=(2, 2), strides=(2, 2), padding='same'))
        representations.append(tf.layers.conv2d(representations[-1], filters=16, kernel_size=5, padding='same', activation=tf.nn.relu))
        representations.append(tf.layers.max_pooling2d(representations[-1], pool_size=(2, 2), strides=(2, 2), padding='same'))
        representations.append(tf.layers.conv2d(representations[-1], filters=1, kernel_size=3, padding='same', activation=tf.nn.relu))
        representations.append(tf.layers.max_pooling2d(representations[-1], pool_size=(2, 2), strides=(2, 2), padding='same'))

        logits = tf.reduce_mean(representations[-1], axis=(1, 2, 3)) # 0 for fake, 1 for real
        representations.append(logits)
        outputs = tf.nn.sigmoid(logits)
        representations.append(outputs)

    vars = tf.trainable_variables(scope='discriminator')

    targets = tf.placeholder(dtype=tf.float32, shape=(None))
    loss = tf.losses.sigmoid_cross_entropy(targets, logits)
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss, var_list = vars)


#%% adversarial training setup
Generator.adversarial_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(Discriminator.targets), Discriminator.logits)
Generator.loss = utils.average([Generator.mse_loss, Generator.adversarial_loss], weights=[1.0, 1.0])
Generator.optimizer = tf.train.AdamOptimizer(1e-3).minimize(Generator.loss, var_list=Generator.vars)


#%% training/loading
session = tf.Session()
saver = tf.train.Saver()
save_location  = 'checkpoints/hybrid.ckpt'

fakes_library = [] # used to prevent instability


def train(num_steps, demo_interval=16, batch_size=16, stored_fakes=1, fakes_batch_size=16):
    global fakes_library
    for i in trange(num_steps):
        if i % demo_interval == 0:
            decoded = session.run(Generator.outputs, feed_dict={Generator.inputs: images.batch(4)[0]})
            images.show(decoded)
        noised_images, real_images = images.batch(batch_size)
        # train discriminator on the batch of real images
        feed = {Discriminator.inputs: real_images, Discriminator.targets: np.ones((batch_size,))}
        session.run(Discriminator.optimizer, feed_dict=feed)
        # train generator on batch & tell discriminator it's seeing fakes
        feed = {Generator.inputs: noised_images, Generator.targets: real_images, Discriminator.targets: np.zeros((batch_size,))}
        new_fakes, _, _ = session.run((Generator.outputs, Generator.optimizer, Discriminator.optimizer), feed_dict=feed)
        # add some newly produced fakes to library
        fakes_library += list(new_fakes[:stored_fakes])
        # train discriminator on some of the fakes stored in the library
        old_fakes = random.choices(fakes_library, k=fakes_batch_size)
        feed = {Discriminator.inputs: old_fakes, Discriminator.targets: np.zeros((batch_size,))}
        session.run(Discriminator.optimizer, feed_dict=feed)


try:
    saver.restore(session, save_location)
except (tf.OpError, ValueError) as error:
    print(f'An error occured: {error}')
    print('Assuming network is to be retrained...')
    session.run(tf.global_variables_initializer())
    train(num_steps=1024)
    saver.save(session, save_location)


#%% test
import os
import skimage
from glob import glob


def test(path, save=False):
    for file in glob(path):
        if not os.path.isfile(file):
            continue
        if file.find('_marbled') >= 0:
            continue
        original = images.to_rgb(skimage.io.imread(file))
        generated = session.run(Generator.outputs, feed_dict={Generator.inputs: [original]})
        images.show(generated, save_as='_marbled.'.join(file.rsplit('.', 1)) if save else None)


test('test/*', save=True)
