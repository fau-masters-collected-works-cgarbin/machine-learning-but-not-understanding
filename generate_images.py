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


def _display_grayscale_image_hex(file: str):
    data = image.imread(file)
    print('Data type: {}, shape: {}'.format(data.dtype, data.shape))
    print('Pixels:')
    for i in range(0, data.shape[0]):
        print(''.join('{:02x}'.format(p) for p in data[i]))


def create_upright_square_dataset():
    """Create a dataset of upright square iamges."""
    SIDE = 10
    for y in range(2, 50, 2):
        for x in range(2, 50, 2):
            right_x = x+SIDE-1
            bottom_y = y+SIDE-1
            _create_square(SQUARE_UPRIGHT_FILE, (x, y), (right_x, y), (x, bottom_y), (right_x, bottom_y))


def test():
    """Test code to check that grayscale images are properly generated.

    It creates one image, then displays the pixel values (in hexadecimal), so we can visually
    inspect that the image type and pixels values are correct.
    """
    coordinates = ((2, 2), (29, 2), (2, 29), (29, 29))
    _create_square(SQUARE_UPRIGHT_FILE, *coordinates)
    _display_grayscale_image_hex(SQUARE_UPRIGHT_FILE.format(*[c for tupl in coordinates for c in tupl]))


if __name__ == "__main__":
    _prepare()
    # test()
    create_upright_square_dataset()
