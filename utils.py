# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import cairo
import numpy as np
from scipy import ndimage
from keras.preprocessing import image
from matplotlib.font_manager import FontProperties
# matplotlib.rc('font', family="Lohit Devanagari")

prop = FontProperties(size=16)
prop.set_file('fonts/Sarun_ThangLuang.ttf')
OUTPUT_DIR = 'image_ocr_LP'

# character classes and matching regex filter
regex = r'^[a-z ]+$'
# alphabet = u'abcdefghijklmnopqrstuvwxyz '
alphabet = u'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮ๑๒๓๔๕๖๗๘๙๐0123456789 '

np.random.seed(55)


# this creates larger "blotches" of noise which look
# more realistic than just adding gaussian noise
# assumes greyscale with pixels ranging from 0 to 1

def speckle(img):
    severity = np.random.uniform(0, 0.6)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    if np.random.rand() > 0.5:
        img_speck = 1 - img_speck
    return img_speck


# paints the string in a random location the bounding box
# also uses a random font, a slight random rotation,
# and a random amount of speckle noise

def paint_text(text, w, h, rotate=False, ud=False, multi_fonts=False):
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    # with cairo.Context(surface) as context:
    context = cairo.Context(surface)
    context.set_source_rgb(1, 1, 1)  # White
    context.paint()
    # this font list works in CentOS 7
    if multi_fonts:
        fonts = ["Sarun's ThangLuang"]
        context.select_font_face(
            np.random.choice(fonts),
            cairo.FONT_SLANT_NORMAL,
            np.random.choice([cairo.FONT_WEIGHT_BOLD, cairo.FONT_WEIGHT_NORMAL]))
    else:
        context.select_font_face("Sarun's ThangLuang",
                                 cairo.FONT_SLANT_NORMAL,
                                 cairo.FONT_WEIGHT_BOLD)
    context.set_font_size(50)
    box = context.text_extents(text)
    border_w_h = (4, 4)
    if box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
        raise IOError(('Could not fit string into image.'
                       'Max char count is too large for given image width.'))

    # teach the RNN translational invariance by
    # fitting text box randomly on canvas, with some room to rotate
    max_shift_x = w - box[2] - border_w_h[0]
    max_shift_y = h - box[3] - border_w_h[1]
    top_left_x = np.random.randint(0, int(max_shift_x))
    top_left_y = h // 2
    context.move_to(0 - int(box[0]), 0 - int(box[1]))
    context.set_source_rgb(0, 0, 0)
    context.show_text(text)

    buf = surface.get_data()
    a = np.frombuffer(buf, np.uint8)
    a.shape = (h, w, 4)
    a = a[:, :, 0]  # grab single channel
    return a, box


def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('`shuffle_mats_or_lists` only supports '
                            'numpy.array and list objects.')
    return ret


# Translation of characters to unique integer values
def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(alphabet.find(char))
    return ret


# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)
