"""Generate training images.

Sources used to write this code:

- https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
- https://note.nkmk.me/en/python-pillow-imagedraw/
- https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/

"""
from PIL import Image, ImageDraw
from matplotlib import image
import os

IMAGE_DIRECTORY = 'images'
# base name foe the images with squares
SQUARE = 'square'
SQUARE_IMAGE = os.path.join(IMAGE_DIRECTORY, SQUARE)


def _prepare():
    os.makedirs(IMAGE_DIRECTORY, exist_ok=True)


def _create_square(top_x: int, top_y: int, width: int, height: int):
    im = Image.new(mode='L', size=(64, 64), color=255)
    draw = ImageDraw.Draw(im)
    draw.rectangle((top_x, top_y, top_x+width-1, top_y+height-1), fill=255, outline=0)
    im.save('{}-{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}.bmp'.format(SQUARE_IMAGE, top_x, top_y, width, height))


def _display_grayscale_image(file: str):
    data = image.imread(file)
    print('Data type: {}, shape: {}'.format(data.dtype, data.shape))
    print('Pixels:')
    for i in range(0, data.shape[0]):
        print(''.join('{:02x}'.format(p) for p in data[i]))


if __name__ == "__main__":
    _prepare()
    _create_square(2, 2, 30, 30)
    _display_grayscale_image('images/square-002-002-030-030.bmp')
