import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.ops import math_ops
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.keras import initializers
from tensorflow.keras import backend as K


class GradientPenalty(Layer):
    def __init__(self, **kwargs):
        super(GradientPenalty, self).__init__(**kwargs)

    def call(self, inputs):
        target, wrt = inputs
        grad = K.gradients(target, wrt)[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))-1

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)


class RandomWeightedAverage(_Merge):
    def build(self, input_shape):
        super(RandomWeightedAverage, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError('A `RandomWeightedAverage` layer should be '
                             'called on exactly 2 inputs')

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `RandomWeightedAverage` layer should be '
                             'called on exactly 2 inputs')

        (x,y) = inputs
        shape = K.shape(x)
        weights = K.random_uniform(shape[:1],0,1)
        for i in range(len(K.int_shape(x))-1):
            weights = K.expand_dims(weights,-1)
        return x*weights + y*(1-weights)


class SNConv2D(Conv2D):

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)
        
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def _get_trainable_var(self):
        if self._trainable_var is None:
            self._trainable_var = K.freezable_variable(
                self._trainable, name=self.name + '_trainable')
        return self._trainable_var

    def _get_training_value(self, training=None):
        if training is None:
            training = K.learning_phase()
        if isinstance(training, int):
            training = bool(training)
        if base_layer_utils.is_in_keras_graph():
            training = math_ops.logical_and(training, self._get_trainable_var())
        else:
            training = math_ops.logical_and(training, self.trainable)
        return training

    def call(self, inputs, training=None):
        training = self._get_training_value(training)

        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)

        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v

        #Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma = K.dot(_v, W_reshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma

        if training == False:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            u = self.u.assign(tf.cond(
                training,
                lambda: _u,
                lambda: self.u
            ))
            with tf.control_dependencies([u]):
                W_bar = K.reshape(W_bar, W_shape)

        outputs = K.conv2d(
                inputs,
                W_bar,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)

        return outputs


class SNDense(Dense):
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn',
                                 trainable=False)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def _get_trainable_var(self):
        if self._trainable_var is None:
            self._trainable_var = K.freezable_variable(
                self._trainable, name=self.name + '_trainable')
        return self._trainable_var

    def _get_training_value(self, training=None):
        if training is None:
            training = K.learning_phase()
        if isinstance(training, int):
            training = bool(training)
        if base_layer_utils.is_in_keras_graph():
            training = math_ops.logical_and(training, self._get_trainable_var())
        else:
            training = math_ops.logical_and(training, self.trainable)
        return training
        
    def call(self, inputs, training=None):
        training = self._get_training_value(training)

        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)

        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
            
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma = K.dot(_v, W_reshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma

        if training == False:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            u = self.u.assign(tf.cond(
                training,
                lambda: _u,
                lambda: self.u
            ))
            with tf.control_dependencies([u]):
                W_bar = K.reshape(W_bar, W_shape)
        
        output = K.dot(inputs, W_bar)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output 


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1,1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (
            s[0], 
            None if s[1] is None else s[1]+2*self.padding[0],
            None if s[2] is None else s[2]+2*self.padding[1],
            s[3]
        )

    def call(self, x):
        (i_pad,j_pad) = self.padding
        return tf.pad(x, [[0,0], [i_pad,i_pad], [j_pad,j_pad], [0,0]], 'REFLECT')
