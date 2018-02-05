#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: fc.py


import tensorflow as tf

from .common import layer_register, rename_get_variable, VariableHolder
from .utils import parse_args
from ..tfutils import symbolic_functions as symbf

__all__ = ['FullyConnected']


@layer_register(log_shape=True)
def FullyConnected(x, *args, **kwargs):
    """
    A wrapper around `tf.layers.Dense`.

    Differences: Default weight initializer is variance_scaling_initializer(2.0).

    Variable Names:

    * ``W``: weights of shape [in_dim, out_dim]
    * ``b``: bias
    """
    tfargs = parse_args(
        args=args,
        kwargs=kwargs,
        args_names=['units'],
        name_mapping={
            'out_dim': 'units',
            'W_init': 'kernel_initializer',
            'b_init': 'bias_initializer'})
    tfargs.setdefault('kernel_initializer', tf.variance_scaling_initializer(2.0))
    tfargs.setdefault('bias_initializer', tf.constant_initializer())

    x = symbf.batch_flatten(x)

    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        layer = tf.layers.Dense(**tfargs)
        ret = layer.apply(x, scope=tf.get_variable_scope())

    ret.variables = VariableHolder(W=layer.kernel)
    if tfargs.get('use_bias', True):
        ret.variables.b = layer.bias
    return tf.identity(ret, name='output')
