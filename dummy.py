from keras import backend as K
from keras.models import Model
from keras.layers import (Conv2D, Input, Dense, Activation,
                          Reshape, Lambda, LSTM, TimeDistributed,
                          BatchNormalization, Permute)

from keras.optimizers import SGD
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


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc_decode(args):
    y_pred, input_length = args
    seq_len = K.squeeze(input_length, axis=-1)

    top_k_decoded, _ = K.ctc_decode(y_pred=y_pred, input_length=seq_len, greedy=True, beam_width=100, top_paths=1)
    return top_k_decoded[0]


HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

if __name__ == '__main__':
    from keras.utils import plot_model
    main_model, training_model, decoder = construct_model(224, 36, 30, 11)  # +unk

    plot_model(main_model, to_file='main_model.png')
    plot_model(training_model, to_file='training_model.png')
