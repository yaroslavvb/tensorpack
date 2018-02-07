#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: conv2d.py


import tensorflow as tf
from .common import layer_register, VariableHolder
from ..tfutils.common import get_tf_version_number
from ..utils.argtools import shape2d, shape4d, get_data_format
from .tflayer import rename_get_variable, parse_args

__all__ = ['Conv2D', 'Deconv2D']


@layer_register(log_shape=True)
def Conv2D(x, *args, split=1, **kwargs):
    """
    A wrapper around `tf.layers.Conv2D`.

    Differences:

    1. Default weight initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'

    Args:
        split (int): Group convolution. Defaults to 1 (no group).
            Note that this is not a fast implementation.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    tfargs = parse_args(
        args=args, kwargs=kwargs,
        args_names=['filters', 'kernel_size'],
        name_mapping={
            'out_channel': 'filters',
            'kernel_shape': 'kernel_size',
            'stride': 'strides',
        }
    )
    tfargs.setdefault('kernel_initializer', tf.variance_scaling_initializer(scale=2.0))
    tfargs.setdefault('bias_initializer', tf.constant_initializer())
    tfargs.setdefault('padding', 'same')

    if split == 1:
        with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
            layer = tf.layers.Conv2D(**tfargs)
            ret = layer.apply(x, scope=tf.get_variable_scope())
            ret = tf.identity(ret, name='output')

        ret.variables = VariableHolder(W=layer.kernel)
        if tfargs.get('use_bias', True):
            ret.variables.b = layer.bias

    else:
        # group conv implementation
        data_format = get_data_format(
            tfargs.get('data_format', 'channels_last'), tfmode=False)
        in_shape = x.get_shape().as_list()
        channel_axis = 3 if data_format == 'NHWC' else 1
        in_channel = in_shape[channel_axis]
        assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
        assert in_channel % split == 0

        out_channel = tfargs.get('filters')
        dilation_rate = tfargs.get('dilation_rate', 1)
        assert out_channel % split == 0
        assert dilation_rate == 1 or get_tf_version_number() >= 1.5, 'TF>=1.5 required for group dilated conv'

        kernel_shape = shape2d(tfargs.get('kernel_size'))
        padding = tfargs.get('padding').upper()
        filter_shape = kernel_shape + [in_channel / split, out_channel]
        stride = shape4d(tfargs.get('strides', 1), data_format=data_format)

        kw_args = dict(data_format=data_format)
        if get_tf_version_number() >= 1.5:
            kw_args['dilations'] = shape4d(dilation_rate, data_format=data_format)

        W = tf.get_variable(
            'W', filter_shape, initializer=tfargs.get('kernel_initializer'))

        use_bias = tfargs.get('use_bias', True)
        if use_bias:
            b = tf.get_variable('b', [out_channel], initializer=tfargs.get('bias_initializer'))

        inputs = tf.split(x, split, channel_axis)
        kernels = tf.split(W, split, 3)
        outputs = [tf.nn.conv2d(i, k, stride, padding, **kw_args)
                   for i, k in zip(inputs, kernels)]
        conv = tf.concat(outputs, channel_axis)
        activation = tfargs.get('activation', tf.identity)
        ret = activation(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')

        ret.variables = VariableHolder(W=W)
        if use_bias:
            ret.variables.b = b
    return ret


@layer_register(log_shape=True)
def Deconv2D(x, *args, **kwargs):
    """
    A wrapper around `tf.layers.Conv2DTranspose`.

    Differences:

    1. Default weight initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """

    tfargs = parse_args(
        args=args, kwargs=kwargs,
        args_names=['filters', 'kernel_size'],
        name_mapping={
            'out_channel': 'filters',
            'kernel_shape': 'kernel_size',
            'stride': 'strides'
        }
    )
    tfargs.setdefault('kernel_initializer', tf.variance_scaling_initializer(scale=2.0))
    tfargs.setdefault('bias_initializer', tf.constant_initializer())
    tfargs.setdefault('padding', 'same')

    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        layer = tf.layers.Conv2DTranspose(**tfargs)
        ret = layer.apply(x, scope=tf.get_variable_scope())

    ret.variables = VariableHolder(W=layer.kernel)
    if tfargs.get('use_bias', True):
        ret.variables.b = layer.bias
    return tf.identity(ret, name='output')
