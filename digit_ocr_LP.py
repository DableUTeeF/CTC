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
    if dense:
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
        l = np.random.randint(nmax//2, high=nmax)
        if l < nmax:
            break

    s, t = '', []
    for j in range(l):
        j = np.random.choice(range(len(alphabet)))-1
        s = s + digit[j]
        t.append(digit[j])

    # ii = text2img(s, face_list[np.random.randint(nface)], np.random.choice([True, False], p=[0.1, 0.9]))
    ii = text2img(s, face_list[np.random.randint(nface)], False)
    ii = degrade(ii)

    oo = cv2.resize(ii, (224, 36), cv2.INTER_AREA)

    return oo, t, len(t)


####################################################################################################################################
from keras import backend as K
from keras.models import Model
from keras.layers import (Conv2D, Input, Dense, Activation,
                          Reshape, Lambda, GRU, MaxPooling2D,
                          concatenate, add)
from keras.optimizers import SGD


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
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
    input_shape = (w, h, 1)
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512
    act = 'relu'

    x = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(16, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(x)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(16, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (w // (pool_size ** 2),
                        (h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True,
                kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True,
                 go_backwards=True, kernel_initializer='he_normal',
                 name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True,
                kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                 kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(11, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    z = Activation('softmax', name='softmax')(inner)

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
    # decoder = K.function([z, input_length], [dec])  # todo: this is the old one
    decoder = K.function([x], [z])

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
        img = img.astype(np.float) / 255.
        # print(img.shape)
        img = np.rollaxis(img, 1)
        batch_x[i, :, :, 0] = img
        batch_y[i, :txtlen] = digit.index(txt)
        batch_ylen[i] = txtlen
        batch_declen[i] = 12  # heuristic!!

    inputs = {'the_input': batch_x,
              'the_labels': batch_y,
              'input_length': batch_declen,
              'label_length': batch_ylen,
              }
    outputs = {'ctc': np.zeros([batch_size, 1])}  # dummy data for dummy loss function

    return training_model.train_on_batch(inputs, outputs)


import itertools
alphabet = u'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮ๑๒๓๔๕๖๗๘๙๐0123456789 '


def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)


def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret


def debug(main_model, decoder, unk,
          digit, face_list, nface, nmax):
    img, txt, txtlen = random_example(digit, face_list, nface, nmax)

    x = img.astype(np.float) / 255.
    # todo: old
    # out = main_model.predict(x.reshape((1, 224, 36, 1)))
    #
    # decoded_sequences = decoder([out, np.ones((out.shape[0], 1)) * 14])
    # decoded_sequences = decoded_sequences[0][0]
    # todo: new
    x = np.rollaxis(x, 1)
    res = decode_batch(decoder, x.reshape((1, 224, 36, 1)))
    outtxt = res
    # for i in res:
    #     if i == unk:
    #         outtxt = outtxt + ' '
    #     else:
    #         outtxt = outtxt + digit[i]

    print(txt, ' VS ', outtxt)
    cv2.imshow("debug", img)
    cv2.waitKey(-1)


if __name__ == '__main__':
    ttflist = 'list.txt'
    face_list = load_font(ttflist)
    nface = len(face_list)

    digit = list(alphabet[:-1])

    main_model, training_model, decoder = construct_model(224, 36, 16, 11)  # todo: Official use max_str_len=16 from 30
    main_model.summary()
    training_model.summary()
    l = 0

    # todo: new
    from digit_ocr_off import DigitImageGenerator

    minibatch_size = 32
    img_w = 224
    img_h = 36
    pool_size = 2
    words_per_epoch = 16000
    val_words = int(words_per_epoch * 0.2)
    img_gen = DigitImageGenerator(
        monogram_file='',
        bigram_file='',
        minibatch_size=minibatch_size,
        img_w=img_w,
        img_h=img_h,
        downsample_factor=(pool_size ** 2),
        val_split=words_per_epoch - val_words)

    # training_model.fit_generator(
    #     generator=img_gen.next_train(),
    #     steps_per_epoch=(words_per_epoch - val_words) // minibatch_size,
    #     epochs=20,
    #     validation_data=img_gen.next_val(),
    #     validation_steps=val_words // minibatch_size,
    #     callbacks=[img_gen],
    #     initial_epoch=0)
    # main_model.save_weights('weights/main_model.h5')
    # training_model.save_weights('weights/training_model.h5')
    # main_model.load_weights('weights/main_model.h5')
    # training_model.load_weights('weights/training_model.h5')
    #
    # main_model.save('weights/main_model_.h5')
    # training_model.save('weights/training_model_.h5')
    img_gen.on_train_begin(0)
    img_gen.on_epoch_begin(0)
    for i in range(5000):
        try:
            nmax = i // 1000 + 3
            loss = update_batch(training_model, 16, nmax,
                                digit, face_list, nface, 10,
                                8)
            # inputs, outputs = img_gen.next_train().__next__()
            # loss = training_model.train_on_batch(inputs, outputs)
            l += loss
            print('\033[96m'f'{i+1}/5000: {l / (i+1)} : {loss}''\033[0m', end='\r')
            # if loss < 1e-4:
            #     break
        except KeyboardInterrupt:
            break
    print()
    for i in range(5):
        debug(main_model, decoder, 5,
              digit, face_list, nface, 5)
