import numpy as np
import os
import itertools
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Lambda
from keras.layers import Reshape, BatchNormalization, GlobalAveragePooling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from PIL import Image
import utils
# alphabet = u'ABCDEFTHIJKLMNOPQRSTUVWXYZ0123456789 '
alphabet = u'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮ๑๒๓๔๕๖๗๘๙๐0123456789 '
os.environ['CUDA_VISIBLE_DEVICES'] = ""


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def decode_batch(test_func, word_batch):
    out, classifier = test_func([word_batch])
    ret = []
    prov = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
        maxidx = np.argmax(classifier[j])
        p = utils.provinces[maxidx]
        prov.append(p)
    return ret, prov


# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)


def get_model(absolute_max_string_len, img_w, img_h=40):
    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 3)

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
    classifier = Conv2D(64, 3, padding='same', activation=act, kernel_initializer='he_normal',
                        name='conv3')(inner)
    classifier = BatchNormalization(name='bn3')(classifier)
    classifier = MaxPooling2D(pool_size=(pool_size, pool_size), name='max3')(classifier)
    classifier = Conv2D(256, 3, padding='same', activation=act, kernel_initializer='he_normal',
                        name='conv4')(classifier)
    classifier = BatchNormalization(name='bn4')(classifier)
    # classifier = MaxPooling2D(pool_size=(pool_size, pool_size), name='max4')(classifier)
    classifier = GlobalAveragePooling2D(name='avg_pool')(classifier)
    classifier = Dense(77, activation='softmax', name='province')(classifier)
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
    inner = Dense(68, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=[y_pred, classifier]).summary()

    labels = Input(name='the_labels',
                   shape=[8], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence

    test_func = K.function([input_data], [y_pred, classifier])

    model = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=[loss_out, classifier])
    model.load_weights('image_ocr_LP/2019:04:29:15:49:10/weights15.h5')

    return test_func


if __name__ == '__main__':
    w, h = 270, 120
    test_func = get_model(7, w, h)
    image = Image.open('replate002.jpg').convert('RGB').resize((w, h))
    im = np.array(image).astype('float32') / 255.
    im = np.rollaxis(im, 1)
    ret = decode_batch(test_func, np.expand_dims(im[..., ::-1], 0))
    print(ret)
    im = Image.fromarray((im*255).astype('uint8'))
    im.show()
    image.show()
