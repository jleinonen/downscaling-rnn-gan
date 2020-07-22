import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, UpSampling2D, Layer
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import TimeDistributed, Lambda

from blocks import res_block
from layers import SNDense, SNConv2D
from layers import ReflectionPadding2D
from rnn import CustomGateGRU


def initial_state_model(num_preproc=3):
    initial_frame_in = Input(shape=(None,None,1))
    noise_in_initial = Input(shape=(None,None,8),
        name="noise_in_initial")

    h = ReflectionPadding2D(padding=(1,1))(initial_frame_in)
    h = Conv2D(256-noise_in_initial.shape[-1], kernel_size=(3,3))(h)
    h = Concatenate()([h,noise_in_initial])
    for i in range(num_preproc):
        h = res_block(256, activation='relu')(h)

    return Model(
        inputs=[initial_frame_in,noise_in_initial],
        outputs=h
    )


def generator(num_channels=1, num_timesteps=8, num_preproc=3):
    initial_state = Input(shape=(None,None,256))
    noise_in_update = Input(shape=(num_timesteps,None,None,8),
        name="noise_in_update")
    lores_in = Input(shape=(num_timesteps,None,None,num_channels),
        name="cond_in")
    inputs = [lores_in, initial_state, noise_in_update]

    xt = TimeDistributed(ReflectionPadding2D(padding=(1,1)))(lores_in)
    xt = TimeDistributed(Conv2D(256-noise_in_update.shape[-1], 
        kernel_size=(3,3)))(xt)
    xt = Concatenate()([xt,noise_in_update])
    for i in range(num_preproc):
        xt = res_block(256, time_dist=True, activation='relu')(xt)

    def gen_gate(activation='sigmoid'):
        def gate(x):
            x = ReflectionPadding2D(padding=(1,1))(x)
            x = Conv2D(256, kernel_size=(3,3))(x)
            if activation is not None:
                x = Activation(activation)(x)
            return x
        return Lambda(gate)
    
    x = CustomGateGRU(
        update_gate=gen_gate(),
        reset_gate=gen_gate(),
        output_gate=gen_gate(activation=None),
        return_sequences=True,
        time_steps=num_timesteps
    )([xt,initial_state])

    h = x[:,-1,...]
    
    block_channels = [256, 256, 128, 64, 32]
    for (i,channels) in enumerate(block_channels):
        if i > 0:
            x = TimeDistributed(UpSampling2D(interpolation='bilinear'))(x)
        x = res_block(channels, time_dist=True, activation='leakyrelu')(x)

    x = TimeDistributed(ReflectionPadding2D(padding=(1,1)))(x)
    img_out = TimeDistributed(Conv2D(num_channels, kernel_size=(3,3),
        activation='sigmoid'))(x)

    model = Model(inputs=inputs, outputs=[img_out,h])

    def noise_shapes(img_shape=(128,128)):
        noise_shape_update = (
            num_timesteps, img_shape[0]//16, img_shape[1]//16, 8
        )
        return [noise_shape_update]

    return (model, noise_shapes)


def generator_initialized(gen, init_model,
    num_channels=1, num_timesteps=8):
    noise_in_initial = Input(shape=(None,None,8),
        name="noise_in_initial")
    noise_in_update = Input(shape=(num_timesteps,None,None,8),
        name="noise_in_update")
    lores_in = Input(shape=(num_timesteps,None,None,num_channels),
        name="cond_in")
    inputs = [lores_in, noise_in_initial, noise_in_update]

    initial_state = init_model([lores_in[:,0,...], noise_in_initial])
    (img_out,h) = gen([lores_in, initial_state, noise_in_update])

    model = Model(inputs=inputs, outputs=img_out)

    def noise_shapes(img_shape=(128,128)):
        noise_shape_initial = (img_shape[0]//16, img_shape[1]//16, 8)
        noise_shape_update = (
            num_timesteps, img_shape[0]//16, img_shape[1]//16, 8
        )
        return [noise_shape_initial, noise_shape_update]

    return (model, noise_shapes)


def generator_deterministic(gen_init, num_channels=1, num_timesteps=8):
    lores_in = Input(shape=(num_timesteps,None,None,num_channels),
        name="cond_in")

    def zeros_noise(input, which):
        shape = tf.shape(input)
        if which == 'init':
            shape = tf.stack([shape[0],shape[1],shape[2],8])
        elif which == 'update':
            shape = tf.stack([shape[0],num_timesteps,shape[1],shape[2],8])
        return tf.fill(shape, 0.0)

    init_zeros = Lambda(lambda x: zeros_noise(x, 'init'))(lores_in)
    update_zeros = Lambda(lambda x: zeros_noise(x, 'update'))(lores_in)
    img_out = gen_init([lores_in, init_zeros, update_zeros])

    model = Model(inputs=lores_in, outputs=img_out)

    return model



def discriminator(num_channels=1, num_timesteps=8):
    hires_in = Input(shape=(num_timesteps,None,None,num_channels), name="sample_in")
    lores_in = Input(shape=(num_timesteps,None,None,num_channels), name="cond_in")

    x_hr = hires_in
    x_lr = lores_in

    block_channels = [32, 64, 128, 256]
    for (i,channels) in enumerate(block_channels):
        x_hr = res_block(channels, time_dist=True,
            norm="spectral", stride=2)(x_hr)
        x_lr = res_block(channels, time_dist=True,
            norm="spectral")(x_lr)

    x_joint = Concatenate()([x_lr,x_hr])
    x_joint = res_block(256, time_dist=True, norm="spectral")(x_joint)
    x_joint = res_block(256, time_dist=True, norm="spectral")(x_joint)

    x_hr = res_block(256, time_dist=True, norm="spectral")(x_hr)
    x_hr = res_block(256, time_dist=True, norm="spectral")(x_hr)    

    def disc_gate(activation='sigmoid'):
        def gate(x):
            x = ReflectionPadding2D(padding=(1,1))(x)
            x = SNConv2D(256, kernel_size=(3,3),
                kernel_initializer='he_uniform')(x)
            if activation is not None:
                x = Activation(activation)(x)
            return x
        return Lambda(gate)

    h = Lambda(lambda x: tf.zeros_like(x[:,0,...]))
    x_joint = CustomGateGRU(
        update_gate=disc_gate(),
        reset_gate=disc_gate(),
        output_gate=disc_gate(activation=None),
        return_sequences=True,
        time_steps=num_timesteps
    )([x_joint,h(x_joint)])
    x_hr = CustomGateGRU(
        update_gate=disc_gate(),
        reset_gate=disc_gate(),
        output_gate=disc_gate(activation=None),
        return_sequences=True,
        time_steps=num_timesteps
    )([x_hr,h(x_hr)])

    x_avg_joint = TimeDistributed(GlobalAveragePooling2D())(x_joint)
    x_avg_hr = TimeDistributed(GlobalAveragePooling2D())(x_hr)

    x = Concatenate()([x_avg_joint,x_avg_hr])
    x = TimeDistributed(SNDense(256))(x)
    x = LeakyReLU(0.2)(x)

    disc_out = TimeDistributed(SNDense(1))(x)

    disc = Model(inputs=[lores_in, hires_in], outputs=disc_out,
        name='disc')

    return disc

