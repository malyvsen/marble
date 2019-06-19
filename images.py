#%% imports
#% matplotlib inline
import skimage
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa
from imageio import mimsave


#%% loading
def to_rgb(image):
    if len(np.shape(image)) == 2:
        return skimage.color.gray2rgb(image)
    return image[:, :, :3]


images = []
for image in skimage.io.imread_collection('statues/*'):
    images.append(to_rgb(image))
images = np.array(images)


#%% augmentation pipeline
augmented_shape = (64, 64)

augmenter = iaa.Sequential([
    iaa.Fliplr(p=0.5),
    iaa.Affine(scale=(0.5, 1.0), rotate=(-5, 5), mode='reflect'),
    iaa.CropToFixedSize(width=augmented_shape[0], height=augmented_shape[1], position='normal'),
    iaa.Resize(size={'height': augmented_shape[0], 'width': 'keep-aspect-ratio'}),
    iaa.Resize(size={'height': 'keep-aspect-ratio', 'width': augmented_shape[1]})
])


noiser = iaa.Sequential([
    iaa.SomeOf(
        (1, 2),
        [iaa.CoarseDropout(p=(0.0, 0.2), size_percent=(0.01, 0.05)), iaa.AdditiveGaussianNoise(scale=(8, 16))],
        random_order=True
    )
])


#%% utils
def batch(size):
    originals = images[np.random.choice(len(images), size)]
    augmented = np.array([augmenter.augment_image(image) for image in originals])
    noised = noiser.augment_images(augmented)
    return noised, augmented


def demo_board(images):
    return np.clip(np.concatenate(images, 1).astype(np.uint8), 0, 255)


def show(images, save_as=None):
    to_show = demo_board(images)
    if save_as is not None:
        skimage.io.imsave(save_as, to_show)
    plt.imshow(to_show)
    plt.show()
    return to_show


def save_gif(path, demo_boards, fps=25):
    mimsave(path, demo_boards, fps=fps)
