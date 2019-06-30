import os
import numpy as np
from skimage.io import imread
from skimage.measure import compare_ssim as ssim



def compare(name, format):
    original = imread(f'{name}.{format}')
    marbled = imread(f'{name}_marbled.{format}')
    cut_dimensions = np.minimum(np.shape(original), np.shape(marbled))
    return ssim(original[:cut_dimensions[0], :cut_dimensions[1]], marbled[:cut_dimensions[0], :cut_dimensions[1]], multichannel=True)


if __name__ == '__main__':
    for filename in os.listdir('./'):
        if (filename.endswith('.jpg') or filename.endswith('.png')) and 'marbled' not in filename:
            image_name = filename[:filename.find('.')]
            format = filename[filename.find('.') + 1:]
            print(f'{image_name.title()}: {compare(image_name, format)}')
