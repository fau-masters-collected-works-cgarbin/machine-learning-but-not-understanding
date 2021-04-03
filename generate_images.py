"""Generate grayscale training images.

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


def _display_grayscale_image_hex(file: str):
    data = image.imread(file)
    print('Data type: {}, shape: {}'.format(data.dtype, data.shape))
    print('Pixels:')
    for i in range(0, data.shape[0]):
        print(''.join('{:02x}'.format(p) for p in data[i]))


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
        _display_grayscale_image_hex(SQUARE_UPRIGHT_FILE.format(*[c for tupl in coordinates for c in tupl]))
    else:
        coordinates = ((2, 11), (5, 2), (9, 11))
        _create_triangle(TRIANGLE_UPRIGHT_FILE, *coordinates)
        _display_grayscale_image_hex(TRIANGLE_UPRIGHT_FILE.format(*[c for tupl in coordinates for c in tupl]))


if __name__ == "__main__":
    _prepare()
    # test(true)
    # test(false)
    create_upright_square_dataset()
    create_upright_triangle_dataset()
