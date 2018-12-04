import ctypes as ct

import cairo
import cv2
import numpy as np
from PIL import Image

_initialized = False


def create_cairo_font_face_for_file(filename, faceindex=0, loadoptions=0):
    """given the name of a font file, and optional faceindex to pass to FT_New_Face" \
    " and loadoptions to pass to cairo_ft_font_face_create_for_ft_face, creates" \
    " a cairo.FontFace object that may be used to render text with that font."""
    global _initialized
    global _freetype_so
    global _cairo_so
    global _ft_lib
    global _ft_destroy_key
    global _surface

    CAIRO_STATUS_SUCCESS = 0
    FT_Err_Ok = 0

    if True:  # not _initialized:
        # find shared objects
        _freetype_so = ct.CDLL("libfreetype.so")
        _cairo_so = ct.CDLL("libcairo.so")
        _cairo_so.cairo_ft_font_face_create_for_ft_face.restype = ct.c_void_p
        _cairo_so.cairo_ft_font_face_create_for_ft_face.argtypes = [ct.c_void_p, ct.c_int]
        _cairo_so.cairo_font_face_get_user_data.restype = ct.c_void_p
        _cairo_so.cairo_font_face_get_user_data.argtypes = (ct.c_void_p, ct.c_void_p)
        _cairo_so.cairo_font_face_set_user_data.argtypes = (ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p)
        _cairo_so.cairo_set_font_face.argtypes = [ct.c_void_p, ct.c_void_p]
        _cairo_so.cairo_font_face_status.argtypes = [ct.c_void_p]
        _cairo_so.cairo_font_face_destroy.argtypes = (ct.c_void_p,)
        _cairo_so.cairo_status.argtypes = [ct.c_void_p]
        # initialize freetype
        _ft_lib = ct.c_void_p()
        status = _freetype_so.FT_Init_FreeType(ct.byref(_ft_lib))
        if status != FT_Err_Ok:
            raise RuntimeError("Error %d initializing FreeType library." % status)

        # end if

        class PycairoContext(ct.Structure):
            _fields_ = \
                [
                    ("PyObject_HEAD", ct.c_byte * object.__basicsize__),
                    ("ctx", ct.c_void_p),
                    ("base", ct.c_void_p),
                ]

        # end PycairoContext

        _surface = cairo.ImageSurface(cairo.FORMAT_A8, 0, 0)
        _ft_destroy_key = ct.c_int()  # dummy address
        _initialized = True
    # end if

    ft_face = ct.c_void_p()
    cr_face = None
    try:
        # load FreeType face
        status = _freetype_so.FT_New_Face(_ft_lib, filename.encode("utf-8"), faceindex, ct.byref(ft_face))
        if status != FT_Err_Ok:
            raise RuntimeError("Error %d creating FreeType font face for %s" % (status, filename))
        # end if

        # create Cairo font face for freetype face
        cr_face = _cairo_so.cairo_ft_font_face_create_for_ft_face(ft_face, loadoptions)
        status = _cairo_so.cairo_font_face_status(cr_face)
        if status != CAIRO_STATUS_SUCCESS:
            raise RuntimeError("Error %d creating cairo font face for %s" % (status, filename))
        # end if
        # Problem: Cairo doesn't know to call FT_Done_Face when its font_face object is
        # destroyed, so we have to do that for it, by attaching a cleanup callback to
        # the font_face. This only needs to be done once for each font face, while
        # cairo_ft_font_face_create_for_ft_face will return the same font_face if called
        # twice with the same FT Face.
        # The following check for whether the cleanup has been attached or not is
        # actually unnecessary in our situation, because each call to FT_New_Face
        # will return a new FT Face, but we include it here to show how to handle the
        # general case.
        if _cairo_so.cairo_font_face_get_user_data(cr_face, ct.byref(_ft_destroy_key)) is None:
            status = _cairo_so.cairo_font_face_set_user_data(
                    cr_face,
                    ct.byref(_ft_destroy_key),
                    ft_face,
                    _freetype_so.FT_Done_Face
                )
            if status != CAIRO_STATUS_SUCCESS:
                raise RuntimeError("Error %d doing user_data dance for %s" % (status, filename))
            # end if
            ft_face = None  # Cairo has stolen my reference
        # end if

        # set Cairo font face into Cairo context
        cairo_ctx = cairo.Context(_surface)
        cairo_t = PycairoContext.from_address(id(cairo_ctx)).ctx
        _cairo_so.cairo_set_font_face(cairo_t, cr_face)
        status = _cairo_so.cairo_font_face_status(cairo_t)
        if status != CAIRO_STATUS_SUCCESS:
            raise RuntimeError("Error %d creating cairo font face for %s" % (status, filename))
        # end if

    finally:
        _cairo_so.cairo_font_face_destroy(cr_face)
        _freetype_so.FT_Done_Face(ft_face)
    # end try

    # get back Cairo font face as a Python object
    face = cairo_ctx.get_font_face()
    return face


# end create_cairo_font_face_for_file

####################################################################################################################################
def load_font(font_filename):
    face_list = []
    with open(font_filename, 'r', encoding='Latin-1') as file:
        for f in file:
            f = f.strip()
            face = create_cairo_font_face_for_file(f)
            face_list.append(face)
    return face_list


####################################################################################################################################
def text2img(text, face, dense):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1000, 200)
    ctx = cairo.Context(surface)

    ctx.rectangle(0, 0, 1000, 200)
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill()

    ctx.set_source_rgb(0, 0, 0)
    ctx.set_font_face(face)
    ctx.set_font_size(50)

    ctx.move_to(0, 100)

    yy0, yy1 = 100, 100

    x0, y0 = ctx.get_current_point()
    h = 0
    if dense:
        ybearing = 0
        height = 0
        for c in text:
            xb, yb, w, h, dxc, dyc = ctx.text_extents(c)

            yo0, yo1 = 100 + yb, 100 + yb + h
            if dxc == 0:  # vertical
                yo1 = yo1 + 5  # !!!

            if yy0 > yo0:
                yy0 = yo0
            if yy1 < yo1:
                yy1 = yo1
            # print(c, ':', yy0, yy1)

            x, y = ctx.get_current_point()

            if dxc == 0:  # vertical
                y = 100 + 2
            else:
                x = x - 2
                y = 100
            ctx.move_to(x, y)
            ctx.show_text(c)

        # print(ybearing, height)

    else:
        xbearing, ybearing, width, height, dxc, dyc = ctx.text_extents(text)
        ctx.show_text(text)
        yy0 = 100 + ybearing
        yy1 = 100

    x1, y1 = ctx.get_current_point()

    im1 = Image.frombuffer("RGBA", (surface.get_width(), surface.get_height()), surface.get_data(), "raw", "RGBA", 0, 1)

    ii = np.array(im1)
    ii = cv2.cvtColor(ii, cv2.COLOR_RGBA2GRAY)
    ii = ii[int(yy0) - 1:int(yy1) + 2, int(x0):int(x1)]

    del ctx

    return ii


def degrade(img):
    # binarize
    img[img[:, :] < 200] = 0
    img[img[:, :] > 200] = 255

    # for x in range(img.shape[0]):
    #     for y in range(img.shape[1]):
    #         if img[x,y]<200:
    #             img[x,y] = 0
    #         else:
    #             img[x,y] = 255
    # degrade
    kernel = np.ones((2, 2), np.uint8)  #
    img = cv2.erode(img, kernel, iterations=1)

    return img


####################################################################################################################################
def random_example(digit, face_list, nface, nmax):
    while True:
        l = 5 + np.random.randint(5)
        if l < nmax:
            break

    s, t = '', []
    for j in range(l):
        j = np.random.randint(10)
        s = s + digit[j]
        t.append(j)

    ii = text2img(s, face_list[np.random.randint(nface)], np.random.choice([True, False], p=[0.1, 0.9]))
    ii = degrade(ii)

    oo = cv2.resize(ii, (224, 36), cv2.INTER_AREA)

    return oo, t, len(t)


####################################################################################################################################
from keras import backend as K
from keras.models import Model
from keras.layers import (Conv2D, Input, Dense, Activation,
                          Reshape, Lambda, LSTM, TimeDistributed,
                          BatchNormalization, Permute)

from keras.optimizers import SGD


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc_decode(args):
    y_pred, input_length = args
    seq_len = K.squeeze(input_length, axis=-1)

    top_k_decoded, _ = K.ctc_decode(y_pred=y_pred, input_length=seq_len, greedy=True, beam_width=100, top_paths=1)
    return top_k_decoded[0]


# hyp: channels last, input_shape = (h,w,1). ex: (36, 224, 1)
# output shape (max_str_len, nchar)
def construct_model(w, h, max_str_len, nchar):
    # construct main model
    x = Input(name='the_input', shape=(w, h, 1))

    y = Conv2D(32, (3, 3), padding='same', strides=(2, 2), data_format='channels_last')(x)  # -> (18, 112, 32)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv2D(32, (3, 3), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(y)  # -> (9, 56, 32)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv2D(32, (3, 3), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    # for digit, we only perform elastic deform in X
    y = Permute((3, 2, 1))(y)  # -> (8, 9, 56)
    y = TimeDistributed(LSTM(28, return_sequences=True))(y)  # -> elastic deformation on w -> (8, 9, 28))
    y = TimeDistributed(LSTM(14, return_sequences=True))(y)  # -> elastic deformation on w -> (8, 9, 28))
    y = Permute((3, 2, 1))(y)  # -> (56, 9, 8)

    y = Reshape((14, 32 * 9))(y)  # -> (56, 9*8)
    z = TimeDistributed(Dense(nchar, activation='softmax'))(y)

    main_model = Model(inputs=[x], outputs=[z])

    labels = Input(name='the_labels', shape=[max_str_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([z, labels, input_length, label_length])

    model = Model(inputs=[x, labels, input_length, label_length], outputs=loss_out)

    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    # additional model for decoding
    dec = Lambda(ctc_decode, output_shape=[None, ], name='decoder')([z, input_length])
    decoder = K.function([z, input_length], [dec])

    return main_model, model, decoder


####################################################################################################################################

def update_batch(training_model,
                 max_str_len, nmax,
                 digit, face_list, nface, unk,
                 batch_size):
    batch_x = np.zeros((batch_size, 224, 36, 1), dtype=np.float)
    batch_y = np.ones((batch_size, max_str_len)) * unk
    batch_ylen = np.zeros((batch_size, 1))
    batch_declen = np.zeros((batch_size, 1))

    for i in range(batch_size):
        img, txt, txtlen = random_example(digit, face_list, nface, nmax)
        img = img.astype(np.float)
        img = np.rollaxis(img, 1)
        batch_x[i, :, :, 0] = img
        batch_y[i, :txtlen] = txt
        batch_ylen[i] = txtlen
        batch_declen[i] = 12  # heuristic!!

    inputs = {'the_input': batch_x,
              'the_labels': batch_y,
              'input_length': batch_declen,
              'label_length': batch_ylen
              }
    outputs = {'ctc': np.zeros([batch_size, 1])}  # dummy data for dummy loss function

    training_model.train_on_batch(inputs, outputs)


def debug(main_model, decoder, unk,
          digit, face_list, nface, nmax):
    img, txt, txtlen = random_example(digit, face_list, nface, nmax)

    x = img.astype(np.float)
    x = np.rollaxis(x, 1)
    out = main_model.predict(x.reshape((1, 224, 36, 1)))

    decoded_sequences = decoder([out, np.ones((out.shape[0], 1)) * 14])
    decoded_sequences = decoded_sequences[0][0]
    outtxt = ''
    for i in decoded_sequences.tolist():
        if i == unk:
            outtxt = outtxt + ' '
        else:
            outtxt = outtxt + digit[i]

    print(txt, ' VS ', outtxt)
    cv2.imshow("debug", img)
    cv2.waitKey(-1)


####################################################################################################################################
####################################################################################################################################
from tqdm import trange
import argparse

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('ttflist', type=str, help="list of ttf")
    # args = parser.parse_args()
    ttflist = 'list.txt'
    face_list = load_font(ttflist)
    nface = len(face_list)

    digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    main_model, training_model, decoder = construct_model(224, 36, 30, 11)  # +unk
    main_model.summary()
    training_model.summary()

    for i in trange(5000):
        update_batch(training_model, 30, 10,
                     digit, face_list, nface, 10,
                     8)

    for i in range(5):
        debug(main_model, decoder, 10,
              digit, face_list, nface, 10)
