"""Generate grayscale images used in the code.

Sources used to write this code:

- https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
- https://note.nkmk.me/en/python-pillow-imagedraw/
- https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/
- https://stackoverflow.com/a/34748617
"""
from PIL import Image, ImageDraw
from matplotlib import image
import os
import shutil
from typing import Tuple
import glob
import numpy as np
from sklearn import utils

# Canvas height/width (all images are squares)
CANVAS_SIZE = 64

BACKGROUND_COLOR_WHITE = 255
BACKGROUND_COLOR_GRAY_LIGHT = 128
BACKGROUND_COLOR_GRAY_DARK = 64

# The directory where the generated images are stored
IMAGE_DIRECTORY = 'images'

SQUARE_FILE_TEMPLATE = '{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}--{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}.bmp'
SQUARE_UPRIGHT = 'square-upright'
SQUARE_UPRIGHT_FILE = '{}{}{}-{}'.format(IMAGE_DIRECTORY, os.path.sep, SQUARE_UPRIGHT, SQUARE_FILE_TEMPLATE)
SQUARE_ROTATED = 'square-rotated'
SQUARE_ROTATED_FILE = '{}{}{}-{}'.format(IMAGE_DIRECTORY, os.path.sep, SQUARE_ROTATED, SQUARE_FILE_TEMPLATE)
SQUARE_GRAY_DARK = 'square-gray-dark'
SQUARE_GRAY_DARK_FILE = '{}{}{}-{}'.format(IMAGE_DIRECTORY, os.path.sep, SQUARE_GRAY_DARK, SQUARE_FILE_TEMPLATE)
SQUARE_GRAY_LIGHT = 'square-gray-light'
SQUARE_GRAY_LIGHT_FILE = '{}{}{}-{}'.format(IMAGE_DIRECTORY, os.path.sep, SQUARE_GRAY_LIGHT, SQUARE_FILE_TEMPLATE)
SQUARE_SIDE = 10

TRIANGLE_FILE_TEMPLATE = '{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}.bmp'
TRIANGLE_UPRIGHT = 'triangle-upright'
TRIANGLE_UPRIGHT_FILE = '{}{}{}-{}'.format(IMAGE_DIRECTORY, os.path.sep, TRIANGLE_UPRIGHT, TRIANGLE_FILE_TEMPLATE)
TRIANGLE_GRAY_DARK = 'triangle-gray-dark'
TRIANGLE_GRAY_DARK_FILE = '{}{}{}-{}'.format(IMAGE_DIRECTORY, os.path.sep, TRIANGLE_GRAY_DARK, TRIANGLE_FILE_TEMPLATE)
TRIANGLE_HEIGHT = 20
TRIANGLE_BASE = 21

# Labels
LABEL_SQUARE = 0
LABEL_TRIANGLE = 1


def _prepare():
    """Prepare the environment to generate pictures.

    Ensure that we start with a clean list of images, without leftovers from previous runs (that
    may be using differen parameters to generate images).
    """
    shutil.rmtree(IMAGE_DIRECTORY, ignore_errors=True)
    os.makedirs(IMAGE_DIRECTORY, exist_ok=True)


def _create_square(base_name: str, corner1: Tuple[int, ...], corner2: Tuple[int, ...],
                   corner3: Tuple[int, ...], corner4: Tuple[int, ...], background: int):
    """Create a square picture and save it.

    All coordinates are in the Pillow system: the top-left corner of the canvas is (0,0).

    Args:
        base_name (str): Base name for the file where the picture will be saved
        corner1..4 (Tuple[int, ...]): [x, y] coordinate of each corner, in order.
        background_color
    """
    im = Image.new(mode='P', size=(CANVAS_SIZE, CANVAS_SIZE), color=background)
    draw = ImageDraw.Draw(im)
    draw.polygon([corner1, corner2, corner3, corner4], outline=0)
    im.save(base_name.format(*corner1, *corner2, *corner3, *corner4))


def _create_triangle(base_name: str, vertex1: Tuple[int, ...], vertex2: Tuple[int, ...],
                     vertex3: Tuple[int, ...], background: int):
    """Create a triangle picture and save it.

    All coordinates are in the Pillow system: the top-left corner of the canvas is (0,0).

    Args:
        base_name (str): Base name for the file where the picture will be saved.
        vertex1..3 (Tuple[int, ...]): [x, y] coordinate of each vertex, order.
    """
    im = Image.new(mode='L', size=(CANVAS_SIZE, CANVAS_SIZE), color=background)
    draw = ImageDraw.Draw(im)
    draw.polygon([vertex1, vertex2, vertex3], outline=0)
    im.save(base_name.format(*vertex1, *vertex2, *vertex3))


def _display_grayscale_image_hex(image_array):
    print('Data type: {}, shape: {}'.format(image_array.dtype, image_array.shape))
    print('Pixels:')
    for i in range(0, image_array.shape[0]):
        print(''.join('{:02x}'.format(p) for p in image_array[i]))


def _create_upright_square_dataset(file_pattern: str, background: int):
    """Create a dataset of upright square iamges.

    The range in the loops are set to have about the same number of squares and triangles.
    """
    for y in range(2, 48, 2):
        for x in range(2, 45, 3):
            right_x = x+SQUARE_SIDE-1
            bottom_y = y+SQUARE_SIDE-1
            _create_square(file_pattern, (x, y), (right_x, y), (right_x, bottom_y), (x, bottom_y),
                           background)


def _create_rotated_square_dataset(file_pattern: str, background: int):
    """Create a dataset of rotated square iamges.

    The range in the loops are set to have about the same number of squares and triangles.
    """
    for y in range(11, 48, 5):
        for x in range(2, 45, 4):
            middle_x = x+SQUARE_SIDE-1
            _create_square(file_pattern, (x, y), (middle_x, y-SQUARE_SIDE+1),
                           (x+2*SQUARE_SIDE-2, y), (middle_x, y+SQUARE_SIDE-1), background)


def _create_upright_triangle_dataset(file_pattern: str, background: int):
    """Create a dataset of upright triangle iamges."""
    half_base = TRIANGLE_BASE // 2
    for y in range(TRIANGLE_HEIGHT+2, 50, 2):
        for x in range(2, 50, 2):
            _create_triangle(file_pattern, (x, y), (x+half_base, y-TRIANGLE_HEIGHT+1),
                             (x+TRIANGLE_BASE-1, y), background)


def _get_images(file_name_pattern: str):
    """Get a list of images from files as a NumPy array."""
    files = glob.glob('{}{}{}*'.format(IMAGE_DIRECTORY, os.path.sep, file_name_pattern))
    images = [image.imread(file) for file in files]
    images_np = np.array(images)
    return images_np


def _get_dataset_from_files(file_name_pattern: str, label: int, test_set_pct: int, shuffle: bool):
    """Create train and test sets from the images in the directory, given the pattern.

    Args:
        file_name_pattern (str): The file name pattern that identifies one set of images.
        label (int): The label to use for the images
        test_set_pct (int): The percentage of images to use for the test set.
        shuffle (bool, optional): Shuffle the images before creating the test set. Defaults to True.
    """
    images_np = _get_images(file_name_pattern)
    if shuffle:
        np.random.shuffle(images_np)

    test_set_size = images_np.shape[0] * test_set_pct // 100
    train_set = images_np[test_set_size:]
    test_set = images_np[:test_set_size]

    train_labels = np.empty(train_set.shape[0], dtype=np.uint8)
    train_labels.fill(label)
    test_labels = np.empty(test_set.shape[0], dtype=np.uint8)
    test_labels.fill(label)

    return (train_set, train_labels), (test_set, test_labels)


def _get_dataset(square_template: str, triangle_template: str, test_set_pct: int, shuffle: bool = True):
    """Create the combined dataset of squares and triangles, based on the given file name templates.

    This code it not very efficient. It create copies of images. It's ok for a small dataset. For
    a larger dataset we may want to work with the file names first and load the images after
    doing all the array operations.

    Args:
        square_template (str): The template to select the files with square images.
        triangle_template (str): The template to select the files with triangle images.
        test_set_pct (int): The percentage of images to use for the test set.
        shuffle (bool, optional): Shuffle the images before creating the test set. Defaults to True.
    """
    (strainset, strainlabel), (stestset, stestlabel) = _get_dataset_from_files(
        square_template, LABEL_SQUARE, test_set_pct, shuffle)
    (ttrainset, ttrainlabel), (ttestset, ttestlabel) = _get_dataset_from_files(
        triangle_template, LABEL_TRIANGLE, test_set_pct, shuffle)

    trainset = np.concatenate((strainset, ttrainset), axis=0)
    trainlabel = np.concatenate((strainlabel, ttrainlabel))
    testset = np.concatenate((stestset, ttestset), axis=0)
    testlabel = np.concatenate((stestlabel, ttestlabel))

    # This is not very efficient because it copies data, but it's ok for a small dataset
    if shuffle:
        trainset, trainlabel = utils.shuffle(trainset, trainlabel)

    return (trainset, trainlabel), (testset, testlabel)


def _test(type: str):
    """Test code to check that grayscale images are properly generated.

    It creates one image, then displays the pixel values (in hexadecimal), so we can visually
    inspect that the image type and pixels values are correct.
    """
    if type == 'square upright':
        coordinates = ((2, 2), (29, 2), (29, 29), (2, 29))
        _create_square(SQUARE_UPRIGHT_FILE, *coordinates, BACKGROUND_COLOR_WHITE)
        im = image.imread(SQUARE_UPRIGHT_FILE.format(*[c for tupl in coordinates for c in tupl]))
        _display_grayscale_image_hex(im)
    elif type == 'square rotated':
        coordinates = ((2, 11), (11, 2), (20, 11), (11, 20))
        _create_square(SQUARE_ROTATED_FILE, *coordinates, BACKGROUND_COLOR_WHITE)
        im = image.imread(SQUARE_ROTATED_FILE.format(*[c for tupl in coordinates for c in tupl]))
        _display_grayscale_image_hex(im)
    elif type == 'triangle upright':
        coordinates = ((2, 11), (5, 2), (9, 11))
        _create_triangle(TRIANGLE_UPRIGHT_FILE, *coordinates, BACKGROUND_COLOR_WHITE)
        im = image.imread(TRIANGLE_UPRIGHT_FILE.format(*[c for tupl in coordinates for c in tupl]))
        _display_grayscale_image_hex(im)


def get_upright_dataset(test_set_pct: int, shuffle: bool = True):
    """Create the combined dataset of upright squares and triangles from the images in the directory.

    Args:
        test_set_pct (int): The percentage of images to use for the test set.
        shuffle (bool, optional): Shuffle the images before creating the test set. Defaults to True.
    """
    return _get_dataset(SQUARE_UPRIGHT, TRIANGLE_UPRIGHT, test_set_pct, shuffle)


def get_square_rotated_dataset():
    """Create the rotated square set from the images in the directory.

    This test set is not meant for training, just for prediction, thus it doesn't have a split nor
    labels.
    """
    return _get_images(SQUARE_ROTATED)


def get_color_dataset(test_set_pct: int, shuffle: bool = True):
    """Create the combined dataset of light gray squares and dark gray triangles from the images in
    the directory.

    Args:
        test_set_pct (int): The percentage of images to use for the test set.
        shuffle (bool, optional): Shuffle the images before creating the test set. Defaults to True.
    """
    return _get_dataset(SQUARE_GRAY_LIGHT, TRIANGLE_GRAY_DARK, test_set_pct, shuffle)


def get_square_dark_gray_dataset():
    """Create the dark gray square set from the images in the directory.

    This test set is not meant for training, just for prediction, thus it doesn't have a split nor
    labels.
    """
    return _get_images(SQUARE_GRAY_DARK)


def get_class_labels():
    """Return the class names indexed by their values."""
    return ['Square', 'Triangle']


def prepare_set(data, labels=None):
    """Prepare a set to be used in training.

    - Normalize the pixel range to 0.0-1.0 to improve training
    - Reshape the dataset to match the neural network input layer
    """

    data = data / 255.0

    data = np.reshape(data, (-1, CANVAS_SIZE, CANVAS_SIZE, 1))

    if labels is not None:
        labels = np.reshape(labels, (-1, 1))

    return data, labels


def create_datasets():
    """Create all datasets."""
    _prepare()

    _create_upright_square_dataset(SQUARE_UPRIGHT_FILE, BACKGROUND_COLOR_WHITE)
    _create_rotated_square_dataset(SQUARE_ROTATED_FILE, BACKGROUND_COLOR_WHITE)
    _create_upright_triangle_dataset(TRIANGLE_UPRIGHT_FILE, BACKGROUND_COLOR_WHITE)

    _create_upright_square_dataset(SQUARE_GRAY_LIGHT_FILE, BACKGROUND_COLOR_GRAY_LIGHT)
    _create_upright_square_dataset(SQUARE_GRAY_DARK_FILE, BACKGROUND_COLOR_GRAY_DARK)
    _create_upright_triangle_dataset(TRIANGLE_GRAY_DARK_FILE, BACKGROUND_COLOR_GRAY_DARK)


if __name__ == "__main__":
    _prepare()

    # _test('square upright')
    # _test('square rotated')
    # _test('triangle upright')

    create_datasets()
