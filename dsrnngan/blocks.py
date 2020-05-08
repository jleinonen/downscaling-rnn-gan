import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Conv2D, Dense, Input
from tensorflow.keras.layers import ELU, LeakyReLU, ReLU, ThresholdedReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, TimeDistributed
from tensorflow.keras.regularizers import l2
from layers import SNConv2D, ReflectionPadding2D


def conv_block(channels, conv_size=(3,3), time_dist=False,
    norm=None, stride=1, activation='leakyrelu', padding='valid'):

    Conv = SNConv2D if norm=="spectral" else Conv2D
    TD = TimeDistributed if time_dist else (lambda x: x)

    def block(x):
        if norm=="batch":
            x = BatchNormalization(momentum=0.8, scale=False)(x)
        if activation == 'leakyrelu':
            x = LeakyReLU(0.2)(x)
        elif activation == 'relu':
            x = ReLU()(x)
        elif activation == 'thresholdedrelu':
            x = ThresholdedReLU()(x)
        elif activation == 'elu':
            x = ELU()(x)
        if padding == 'reflect':
            pad = tuple((s-1)//2 for s in conv_size)
            x = TD(ReflectionPadding2D(padding=pad))(x)
        x = TD(Conv(channels, conv_size, 
            padding='valid' if padding=='reflect' else padding,
            strides=(stride,stride), 
            kernel_regularizer=(l2(1e-4) if norm!="spectral" else None)
        ))(x)
        return x

    return block


def res_block(channels, conv_size=(3,3), stride=1, norm=None,
    time_dist=False, activation='leakyrelu'):

    TD = TimeDistributed if time_dist else (lambda x: x)

    def block(x):
        in_channels = int(x.shape[-1])
        x_in = x
        if (stride > 1):
            x_in = TD(AveragePooling2D(pool_size=(stride,stride)))(x_in)
        if (channels != in_channels):
            x_in = conv_block(channels, conv_size=(1,1), stride=1, 
                activation=False, time_dist=time_dist)(x_in)

        x = conv_block(channels, conv_size=conv_size, stride=stride,
            padding='reflect', norm=norm, time_dist=time_dist,
            activation=activation)(x)
        x = conv_block(channels, conv_size=conv_size, stride=1,
            padding='reflect', norm=norm, time_dist=time_dist,
            activation=activation)(x)

        x = Add()([x,x_in])

        return x

    return block
