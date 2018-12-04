import digit_ocr
import cv2
from digit_ocr_off import paint_text, DigitImageGenerator, ctc_lambda_func, decode_batch2
import os
from keras import backend as K
from keras.layers import Input, Conv2D, concatenate, GRU, MaxPooling2D, Reshape, Dense, add, Activation, Lambda
from keras.optimizers import SGD
from keras.models import Model
import numpy as np


def model_(img_w):
    img_h = 64
    words_per_epoch = 16000
    val_split = 0.2
    val_words = int(words_per_epoch * val_split)

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512
    minibatch_size = 32

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    img_gen = DigitImageGenerator(
        monogram_file=os.path.join('', 'wordlist_mono_clean.txt'),
        bigram_file=os.path.join('', 'wordlist_bi_clean.txt'),
        minibatch_size=minibatch_size,
        img_w=img_w,
        img_h=img_h,
        downsample_factor=(pool_size ** 2),
        val_split=words_per_epoch - val_words)
    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2),
                        (img_h // (pool_size ** 2)) * conv_filters)
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
    inner = Dense(img_gen.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels',
                   shape=[img_gen.absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    test_func = K.function([input_data], [y_pred])
    return model, test_func


if __name__ == '__main__':
    # main_model, training_model, decoder = digit_ocr.construct_model(128, 64, 16, 11)
    # main_model.load_weights('weights/main_model.h5')
    # training_model.load_weights('digit_ocr/2018:11:28:11:21:40/weights05.h5')
    img_w = 224
    model, test_func = model_(img_w)
    model.load_weights('digit_ocr/2018:11:29:10:35:29/weights02.h5')
    img = cv2.imread('misc/68215.png')
    img = cv2.resize(img, (img_w, 64))[:, :, 0]
    img = img.astype('float32') / 255.
    img = np.rollaxis(img, 1)
    cv2.imshow('test', img)
    cv2.waitKey(0)

    print(img.shape)
    res = decode_batch2(test_func, img.reshape((1, *img.shape, 1)))
    print(res)
    ret = paint_text('68215', img_w, 64)
    ret = np.rollaxis(ret, 2, 1)
    res = decode_batch2(test_func, ret.reshape((*ret.shape, 1)))
    print(res)
    cv2.imshow('test', ret[0])
    cv2.waitKey(0)
