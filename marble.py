#%% imports
from tqdm import trange
import tensorflow as tf
import numpy as np
import random
import skimage
import images
import utils


#%% generator architecture
class Generator:
    with tf.variable_scope('encoder'):
        inputs = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))

        representations = [inputs]
        representations.append(tf.layers.conv2d(representations[-1], filters=32, kernel_size=5, padding='same', activation=tf.nn.relu))
        representations.append(utils.pool(representations[-1], 'MAX', pool_size=2))
        representations.append(tf.layers.conv2d(representations[-1], filters=32, kernel_size=3, padding='same', activation=tf.nn.relu))
        representations.append(utils.pool(representations[-1], 'MAX', pool_size=2))
        representations.append(tf.layers.conv2d(representations[-1], filters=16, kernel_size=3, padding='same', activation=tf.nn.relu))
        representations.append(utils.pool(representations[-1], 'MAX', pool_size=2))
        skip = utils.pool(inputs, 'AVG', pool_size=8) # skip connection directly to pixels
        skip_appended = tf.concat((representations[-1], skip), axis=-1)
        representations.append(tf.layers.conv2d(representations[-1], filters=16, kernel_size=3, padding='same', activation=tf.nn.relu))
        representations.append(utils.pool(representations[-1], 'MAX', pool_size=2))

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
    ssim_loss = -tf.reduce_mean(tf.image.ssim(outputs, targets, max_val=255)) / 2 # also normalized into interval of size 1
    # training definition continued after discriminator is initialized


#%% discriminator architecture
class Discriminator:
    with tf.variable_scope('discriminator'):
        inputs = Generator.outputs # will be fed directly if it's a real image

        micro = [inputs] # for errors in small scale
        wide_micro = tf.layers.conv2d(micro[-1], filters=8, kernel_size=(7, 3), padding='same', activation=tf.nn.relu)
        high_micro = tf.layers.conv2d(micro[-1], filters=8, kernel_size=(3, 7), padding='same', activation=tf.nn.relu)
        micro.append(tf.concat((wide_micro, high_micro), axis=-1))
        micro.append(utils.pool(micro[-1], 'MAX', pool_size=4))
        micro.append(tf.layers.conv2d(micro[-1], filters=4, kernel_size=5, padding='same', activation=tf.nn.relu))

        macro = [utils.pool(inputs, 'AVG', pool_size=4)] # for errors in large scale
        wide_macro = tf.layers.conv2d(macro[-1], filters=8, kernel_size=(7, 3), padding='same', activation=tf.nn.relu)
        high_macro = tf.layers.conv2d(macro[-1], filters=8, kernel_size=(3, 7), padding='same', activation=tf.nn.relu)
        macro.append(tf.concat((wide_macro, high_macro), axis=-1))
        macro.append(utils.pool(macro[-1], 'MAX', pool_size=4))
        macro.append(tf.layers.conv2d(macro[-1], filters=4, kernel_size=5, padding='same', activation=tf.nn.relu))

        logits = tf.reduce_mean(micro[-1], axis=(1, 2, 3)) + tf.reduce_mean(macro[-1], axis=(1, 2, 3))
        outputs = tf.nn.sigmoid(logits) # 0 for fake, 1 for real

    vars = tf.trainable_variables(scope='discriminator')

    targets = tf.placeholder(dtype=tf.float32, shape=(None))
    loss = tf.losses.sigmoid_cross_entropy(targets, logits)
    optimizer = tf.train.AdamOptimizer(5e-4).minimize(loss, var_list = vars)


#%% adversarial training setup
Generator.adversarial_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(Discriminator.targets), Discriminator.logits)
Generator.mse_weight = tf.placeholder(tf.float32, shape=())
Generator.ssim_weight = tf.placeholder(tf.float32, shape=())
Generator.adversarial_weight = tf.placeholder(tf.float32, shape=())
Generator.loss = utils.average([Generator.mse_loss, Generator.ssim_loss, Generator.adversarial_loss], weights=[Generator.mse_weight, Generator.ssim_weight, Generator.adversarial_weight])
Generator.optimizer = tf.train.AdamOptimizer(5e-4).minimize(Generator.loss, var_list=Generator.vars)


#%% training/loading
session = tf.Session()
saver = tf.train.Saver()
save_location  = 'checkpoints/hybrid.ckpt'

fakes_library = [] # used to prevent instability
demo_inputs = [skimage.io.imread('test/elon.jpg')]
demo_outputs = []


def train(num_steps, loss_weighter, demo_interval=64, batch_size=16, fakes_batch_size=16, max_stored_fakes=64):
    global fakes_library, demo_inputs, demo_outputs

    for step in trange(num_steps):
        if (step + 1) % demo_interval == 0:
            decoded = session.run(Generator.outputs, feed_dict={Generator.inputs: demo_inputs})
            demo_outputs.append(images.demo_board(decoded))
            images.show(decoded)
        noised_images, real_images = images.batch(batch_size)
        # train discriminator on the batch of real images
        feed = {Discriminator.inputs: real_images, Discriminator.targets: np.ones((batch_size,))}
        session.run(Discriminator.optimizer, feed_dict=feed)
        # train generator on batch & tell discriminator it's seeing fakes
        feed = {Generator.inputs: noised_images, Generator.targets: real_images, Discriminator.targets: np.zeros((batch_size,))}
        feed.update(loss_weighter(step / (num_steps - 1)))
        new_fakes, _, _ = session.run((Generator.outputs, Generator.optimizer, Discriminator.optimizer), feed_dict=feed)
        # add some newly produced fakes to library
        fakes_library += list(new_fakes[:1])
        if len(fakes_library) > max_stored_fakes:
            del fakes_library[random.randrange(len(fakes_library))] # prevent overfilling memory
        # train discriminator on some of the fakes stored in the library
        old_fakes = random.choices(fakes_library, k=fakes_batch_size)
        feed = {Discriminator.inputs: old_fakes, Discriminator.targets: np.zeros((batch_size,))}
        session.run(Discriminator.optimizer, feed_dict=feed)


def loss_weighter(progress):
    return {
        Generator.mse_weight: np.interp(progress, (0, .5), (1, 0)),
        Generator.ssim_weight: np.interp(progress, (0, .5), (0, 2)),
        Generator.adversarial_weight: np.interp(progress, (0, 0.5, 1), (0, 1, 2))}


try:
    saver.restore(session, save_location)
except (tf.OpError, ValueError) as error:
    print(f'An error occured: {error}')
    print('Assuming network is to be retrained...')
    session.run(tf.global_variables_initializer())
    train(num_steps=4096, loss_weighter=loss_weighter)
    saver.save(session, save_location)
    images.save_gif('./training.gif', demo_outputs)


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
