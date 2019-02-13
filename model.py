from keras.models import Model
from keras.layers import Input, Convolution2D, Flatten, Dense, Dropout, ELU, Lambda
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
import keras.backend as K
from config import *
from load_data import generate_data_batch, split_train_val
import os

def get_nvidia_model(summary=True):
    """
    Get the keras Model corresponding to the NVIDIA architecture described in:
    Bojarski, Mariusz, et al. "End to end learning for self-driving cars."

    The paper describes the network architecture but doesn't go into details for some aspects.
    Input normalization, as well as ELU activations are just my personal implementation choice.

    :param summary: show model summary
    :return: keras Model of NVIDIA architecture
    """
    init = 'glorot_uniform'

    if K.backend() == 'theano':
        input_frame = Input(shape=(CONFIG['input_channels'], NVIDIA_H, NVIDIA_W))
    else:
        input_frame = Input(shape=(NVIDIA_H, NVIDIA_W, CONFIG['input_channels']))

    # standardize input
    x = Lambda(lambda z: z / 127.5 - 1.)(input_frame)

    x = Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), init=init)(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), init=init)(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), init=init)(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', init=init)(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', init=init)(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)

    x = Dense(100, init=init)(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Dense(50, init=init)(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Dense(10, init=init)(x)
    x = ELU()(x)
    out = Dense(1, init=init)(x)

    model = Model(input=input_frame, output=out)

    if summary:
        model.summary()

    return model


if __name__ == '__main__':
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    # split udacity csv data into training and validation
    train_data, val_data = split_train_val(csv_driving_data='data_example/data/driving_log.csv')

    # get network model and compile it (default Adam opt)
    nvidia_net = get_nvidia_model(summary=True)
    nvidia_net.compile(optimizer='adam', loss='mse')

    # json dump of model architecture
    with open('logs/model.json', 'w') as f:
        f.write(nvidia_net.to_json())

    # define callbacks to save history and weights
    checkpointer = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5')
    logger = CSVLogger(filename='logs/history.csv')
    tflogger = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=CONFIG['batchsize'], write_graph=True, write_grads=False,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    print('steps per epoch:', len(train_data)/CONFIG['batchsize'])
    # start the training
    nvidia_net.fit_generator(generator=generate_data_batch(train_data, data_dir='data_example/data',augment_data=True, bias=CONFIG['bias']),
                             steps_per_epoch=len(train_data)/CONFIG['batchsize'],
                             epochs=50,
                             validation_data=generate_data_batch(val_data, data_dir='data_example/data', augment_data=False, bias=1.0),
                             validation_steps=len(val_data)/CONFIG['batchsize'],
                             callbacks=[checkpointer, logger, tflogger])
