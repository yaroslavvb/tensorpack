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
def Conv2D(x, out_channel, kernel_shape,
           padding='SAME', stride=1,
           W_init=None, b_init=None,
           activation=tf.identity, split=1, use_bias=True,
           data_format='channels_last', dilation_rate=1):
    """
    2D convolution on 4D inputs.

    Args:
        x (tf.Tensor): a 4D tensor.
            Must have known number of channels, but can have other unknown dimensions.
        out_channel (int): number of output channel.
        kernel_shape: (h, w) tuple or a int.
        stride: (h, w) tuple or a int.
        padding (str): 'valid' or 'same'. Case insensitive.
        split (int): Split channels as used in Alexnet. Defaults to 1 (no split).
        W_init: initializer for W. Defaults to `variance_scaling_initializer(2.0)`, i.e. kaiming-normal.
        b_init: initializer for b. Defaults to zero.
        use_bias (bool): whether to use bias.
        dilation_rate: (h, w) tuple or a int.

    Returns:
        tf.Tensor named ``output`` with attribute `variables`.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    data_format = get_data_format(data_format, tfmode=False)
    in_shape = x.get_shape().as_list()
    channel_axis = 3 if data_format == 'NHWC' else 1
    in_channel = in_shape[channel_axis]
    assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
    assert in_channel % split == 0
    assert out_channel % split == 0
    assert dilation_rate == 1 or get_tf_version_number() >= 1.5, 'TF ver. 1.5 or greater required for dilations'

    kernel_shape = shape2d(kernel_shape)
    padding = padding.upper()
    filter_shape = kernel_shape + [in_channel / split, out_channel]
    stride = shape4d(stride, data_format=data_format)

    kw_args = dict(data_format=data_format)
    if get_tf_version_number() >= 1.5:
        kw_args['dilations'] = shape4d(dilation_rate, data_format=data_format)

    if W_init is None:
        W_init = tf.variance_scaling_initializer(scale=2.0)
    if b_init is None:
        b_init = tf.constant_initializer()

    W = tf.get_variable('W', filter_shape, initializer=W_init)

    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=b_init)

    if split == 1:
        conv = tf.nn.conv2d(x, W, stride, padding, **kw_args)
    else:
        inputs = tf.split(x, split, channel_axis)
        kernels = tf.split(W, split, 3)
        outputs = [tf.nn.conv2d(i, k, stride, padding, **kw_args)
                   for i, k in zip(inputs, kernels)]
        conv = tf.concat(outputs, channel_axis)

    ret = activation(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')
    ret.variables = VariableHolder(W=W)
    if use_bias:
        ret.variables.b = b
    return ret


@layer_register(log_shape=True)
def Deconv2D(x, *args, **kwargs):
    """
    A wrapper around `tf.layers.Conv2DTranspose`.

    Differences: Default weight initializer is variance_scaling_initializer(2.0).

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """

    tfargs = parse_args(
        args=args, kwargs=kwargs,
        args_names=['filters', 'kernel_size'],
        name_mapping={
            'stride': 'strides',
            'W_init': 'kernel_initializer',
            'b_init': 'bias_initializer'
        }
    )
    tfargs.setdefault('kernel_initializer', tf.variance_scaling_initializer(scale=2.0))
    tfargs.setdefault('bias_initializer', tf.constant_initializer())

    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        layer = tf.layers.Conv2DTranspose(**tfargs)
        ret = layer.apply(x, scope=tf.get_variable_scope())

    ret.variables = VariableHolder(W=layer.kernel)
    if tfargs.get('use_bias', True):
        ret.variables.b = layer.bias
    return tf.identity(ret, name='output')
