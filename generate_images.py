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
from typing import Tuple

# The directory where the generated images are stored
IMAGE_DIRECTORY = 'images'
# The template for the file names: type, top-left, top-right, bottom-left, bottom-right
IMAGE_FILE_NAME_TEMPLATE = '{}-{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}--{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}.bmp'
# Image height/width (all images are squares)
IMAGE_SIZE = 64

SQUARE = 'square'
SQUARE_IMAGE = os.path.join(IMAGE_DIRECTORY, SQUARE)


def _prepare():
    """Prepare the environment to generate pictures."""
    os.makedirs(IMAGE_DIRECTORY, exist_ok=True)


def _create_square(top_left: Tuple[int, ...], top_right: Tuple[int, ...],
                   bottom_left: Tuple[int, ...], bottom_right: Tuple[int, ...]):
    """Create a square picture and save it.

    Args:
        top_left (Tuple[int, ...]): [x, y] Coordinate for the top-left corner.
        top_right (Tuple[int, ...]): [x, y] Coordinate for the top-right corner.
        bottom_left (Tuple[int, ...]): [x, y] Coordinate for the bottom-left corner.
        bottom_right (Tuple[int, ...]): [x, y] Coordinate for the bottom-right corner.
    """
    im = Image.new(mode='L', size=(IMAGE_SIZE, IMAGE_SIZE), color=255)
    draw = ImageDraw.Draw(im)
    draw.polygon([top_left, top_right, bottom_right, bottom_left], outline=0)
    im.save(IMAGE_FILE_NAME_TEMPLATE.format(
        SQUARE_IMAGE, *top_left, *top_right, *bottom_left, *bottom_right))


def _display_grayscale_image(file: str):
    data = image.imread(file)
    print('Data type: {}, shape: {}'.format(data.dtype, data.shape))
    print('Pixels:')
    for i in range(0, data.shape[0]):
        print(''.join('{:02x}'.format(p) for p in data[i]))


if __name__ == "__main__":
    _prepare()
    _create_square((2, 2), (29, 2), (2, 29), (29, 29))
    _display_grayscale_image(IMAGE_FILE_NAME_TEMPLATE.format(
        SQUARE_IMAGE, *(2, 2), *(29, 2), *(2, 29), *(29, 29)))
