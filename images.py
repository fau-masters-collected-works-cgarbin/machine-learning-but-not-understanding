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

# The directory where the generated images are stored
IMAGE_DIRECTORY = 'images'
# Canvas height/width (all images are squares)
CANVAS_SIZE = 64
BACKGROUND_COLOR = 255

SQUARE_UPRIGHT = 'square-upright'
SQUARE_UPRIGHT_FILE = '{}{}{}-{}'.format(IMAGE_DIRECTORY, os.path.sep, SQUARE_UPRIGHT,
                                         '{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}--{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}.bmp')
SQUARE_SIDE = 10

TRIANGLE_UPRIGHT = 'triangle-upright'
TRIANGLE_UPRIGHT_FILE = '{}{}{}-{}'.format(IMAGE_DIRECTORY, os.path.sep, TRIANGLE_UPRIGHT,
                                           '{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}.bmp')
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


def _create_square(base_name: str, top_left: Tuple[int, ...], top_right: Tuple[int, ...],
                   bottom_left: Tuple[int, ...], bottom_right: Tuple[int, ...]):
    """Create a square picture and save it.

    All coordinates are in the Pillow system: the top-left corner of the canvas is (0,0).

    Args:
        base_name (str): Base name for the file where the picture will be saved
        top_left (Tuple[int, ...]): [x, y] coordinate for the top-left corner.
        top_right (Tuple[int, ...]): [x, y] coordinate for the top-right corner.
        bottom_left (Tuple[int, ...]): [x, y] coordinate for the bottom-left corner.
        bottom_right (Tuple[int, ...]): [x, y] coordinate for the bottom-right corner.
    """
    im = Image.new(mode='L', size=(CANVAS_SIZE, CANVAS_SIZE), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(im)
    draw.polygon([top_left, top_right, bottom_right, bottom_left], outline=0)
    im.save(base_name.format(*top_left, *top_right, *bottom_left, *bottom_right))


def _create_triangle(base_name: str, vertex1: Tuple[int, ...], vertex2: Tuple[int, ...], vertex3: Tuple[int, ...]):
    """Create a triangle picture and save it.

    All coordinates are in the Pillow system: the top-left corner of the canvas is (0,0).

    Args:
        base_name (str): Base name for the file where the picture will be saved
        vertex1..3 (Tuple[int, ...]): [x, y] coordinate of each vertex
    """
    im = Image.new(mode='L', size=(CANVAS_SIZE, CANVAS_SIZE), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(im)
    draw.polygon([vertex1, vertex2, vertex3], outline=0)
    im.save(base_name.format(*vertex1, *vertex2, *vertex3))


def _display_grayscale_image_hex(image_array):
    print('Data type: {}, shape: {}'.format(image_array.dtype, image_array.shape))
    print('Pixels:')
    for i in range(0, image_array.shape[0]):
        print(''.join('{:02x}'.format(p) for p in image_array[i]))


def create_upright_square_dataset():
    """Create a dataset of upright square iamges.

    The range in the loops are set to have about the same number of squares and triangles.
    """
    for y in range(2, 48, 2):
        for x in range(2, 45, 3):
            right_x = x+SQUARE_SIDE-1
            bottom_y = y+SQUARE_SIDE-1
            _create_square(SQUARE_UPRIGHT_FILE, (x, y), (right_x, y), (x, bottom_y), (right_x, bottom_y))


def create_upright_triangle_dataset():
    """Create a dataset of upright triangle iamges."""
    half_base = TRIANGLE_BASE // 2
    for y in range(TRIANGLE_HEIGHT+2, 50, 2):
        for x in range(2, 50, 2):
            _create_triangle(TRIANGLE_UPRIGHT_FILE, (x, y), (x+half_base, y-TRIANGLE_HEIGHT+1),
                             (x+TRIANGLE_BASE-1, y))


def test(square: bool):
    """Test code to check that grayscale images are properly generated.

    It creates one image, then displays the pixel values (in hexadecimal), so we can visually
    inspect that the image type and pixels values are correct.
    """
    if square:
        coordinates = ((2, 2), (29, 2), (2, 29), (29, 29))
        _create_square(SQUARE_UPRIGHT_FILE, *coordinates)
        im = image.imread(SQUARE_UPRIGHT_FILE.format(*[c for tupl in coordinates for c in tupl]))
        _display_grayscale_image_hex(im)
    else:
        coordinates = ((2, 11), (5, 2), (9, 11))
        _create_triangle(TRIANGLE_UPRIGHT_FILE, *coordinates)
        im = image.imread(TRIANGLE_UPRIGHT_FILE.format(*[c for tupl in coordinates for c in tupl]))
        _display_grayscale_image_hex(im)


def _get_dataset(file_name_pattern: str, label: int, test_set_pct: int, shuffle: bool):
    """Create train and test sets from the images in the directory, given the pattern.

    Args:
        file_name_pattern (str): The file name pattern that identifies one set of images.
        label (int): The label to use for the images
        test_set_pct (int): The percentage of images to use for the test set.
        shuffle (bool, optional): Shuffle the images before creating the test set. Defaults to True.
    """
    files = glob.glob('{}{}{}*'.format(IMAGE_DIRECTORY, os.path.sep, file_name_pattern))
    images = [image.imread(file) for file in files]
    images_np = np.array(images)
    if shuffle:
        np.random.shuffle(images_np)

    test_set_size = len(images_np) * test_set_pct // 100
    train_set = images_np[test_set_size:]
    test_set = images_np[:test_set_size]

    train_labels = np.empty(train_set.shape[0], dtype=np.uint8)
    train_labels.fill(label)
    test_labels = np.empty(test_set.shape[0], dtype=np.uint8)
    test_labels.fill(label)

    return (train_set, train_labels), (test_set, test_labels)


def get_square_upright_dataset(test_set_pct: int, shuffle: bool = True):
    """Create the upright square train and test sets from the images in the directory.

    Args:
        test_set_pct (int): The percentage of images to use for the test set.
        shuffle (bool, optional): Shuffle the images before creating the test set. Defaults to True.
    """
    return _get_dataset(SQUARE_UPRIGHT, LABEL_SQUARE, test_set_pct, shuffle)


def get_triangle_upright_dataset(test_set_pct: int, shuffle: bool = True):
    """Create the upright triangle train and test sets from the images in the directory.

    Args:
        test_set_pct (int): The percentage of images to use for the test set.
        shuffle (bool, optional): Shuffle the images before creating the test set. Defaults to True.
    """
    return _get_dataset(TRIANGLE_UPRIGHT, LABEL_TRIANGLE, test_set_pct, shuffle)


def get_upright_dataset(test_set_pct: int, shuffle: bool = True):
    """Create the combined dataset of upright squares and triangles.

    This code it no t very efficient. It create copies of images. It's ok for a small dataset. For
    a larger dataset we may want to work with the file names first and load the images after
    doing all the array operations.

    Args:
        test_set_pct (int): The percentage of images to use for the test set.
        shuffle (bool, optional): Shuffle the images before creating the test set. Defaults to True.
    """
    (strainset, strainlabel), (stestset, stestlabel) = get_square_upright_dataset(test_set_pct, shuffle)
    (ttrainset, ttrainlabel), (ttestset, ttestlabel) = get_triangle_upright_dataset(test_set_pct, shuffle)

    trainset = np.concatenate((strainset, ttrainset), axis=0)
    trainlabel = np.concatenate((strainlabel, ttrainlabel))
    testset = np.concatenate((stestset, ttestset), axis=0)
    testlabel = np.concatenate((stestlabel, ttestlabel))

    # This is not very efficient because it copies data, but it's ok for a small dataset
    if shuffle:
        trainset, trainlabel = utils.shuffle(trainset, trainlabel)

    return (trainset, trainlabel), (testset, testlabel)


if __name__ == "__main__":
    _prepare()
    # test(True)
    # test(False)
    create_upright_square_dataset()
    create_upright_triangle_dataset()

    # (trs, trl), (tss, tsl) = get_square_upright_dataset(10)
    # _display_grayscale_image_hex(trs[0])
    # (trs, trl), (tss, tsl) = get_triangle_upright_dataset(10)
    # _display_grayscale_image_hex(trs[0])

    (trs, trl), (tss, tsl) = get_upright_dataset(10)
