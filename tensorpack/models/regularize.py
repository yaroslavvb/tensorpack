# -*- coding: UTF-8 -*-
# File: regularize.py


import tensorflow as tf
import re

from ..utils import logger
from ..utils.argtools import graph_memoized
from ..tfutils.tower import get_current_tower_context
from .common import layer_register
from .tflayer import parse_args

__all__ = ['regularize_cost', 'l2_regularizer', 'l1_regularizer', 'Dropout']


@graph_memoized
def _log_once(msg):
    logger.info(msg)


l2_regularizer = tf.contrib.layers.l2_regularizer
l1_regularizer = tf.contrib.layers.l1_regularizer


def regularize_cost(regex, func, name='regularize_cost'):
    """
    Apply a regularizer on trainable variables matching the regex, and print
    the matched variables (only print once in multi-tower training).
    In replicated mode, it will only regularize variables within the current tower.

    Args:
        regex (str): a regex to match variable names, e.g. "conv.*/W"
        func: the regularization function, which takes a tensor and returns a scalar tensor.
            E.g., ``tf.contrib.layers.l2_regularizer``.

    Returns:
        tf.Tensor: the total regularization cost.

    Example:
        .. code-block:: python

            cost = cost + regularize_cost("fc.*/W", l2_regularizer(1e-5))
    """
    assert len(regex)
    ctx = get_current_tower_context()
    if not ctx.is_training:
        # Currently cannot build the wd_cost correctly at inference,
        # because ths vs_name used in inference can be '', therefore the
        # variable filter will fail
        return tf.constant(0, dtype=tf.float32, name='empty_' + name)

    # If vars are shared, regularize all of them
    # If vars are replicated, only regularize those in the current tower
    if ctx.has_own_variables:
        params = ctx.get_collection_in_tower(tf.GraphKeys.TRAINABLE_VARIABLES)
    else:
        params = tf.trainable_variables()

    names = []

    with tf.name_scope(name + '_internals'):
        costs = []
        for p in params:
            para_name = p.op.name
            if re.search(regex, para_name):
                costs.append(func(p))
                names.append(p.name)
        if not costs:
            return tf.constant(0, dtype=tf.float32, name='empty_' + name)

    # remove tower prefix from names, and print
    if len(ctx.vs_name):
        prefix = ctx.vs_name + '/'
        prefixlen = len(prefix)

        def f(name):
            if name.startswith(prefix):
                return name[prefixlen:]
            return name
        names = list(map(f, names))
    logger.info("regularize_cost() found {} tensors.".format(len(names)))
    _log_once("Applying regularizer for {}".format(', '.join(names)))

    return tf.add_n(costs, name=name)


def regularize_cost_from_collection(name='regularize_cost'):
    """
    Get the cost from the regularizers in ``tf.GraphKeys.REGULARIZATION_LOSSES``.
    In replicated mode, will only regularize variables within the current tower.

    Returns:
        a scalar tensor, the regularization loss, or None
    """
    ctx = get_current_tower_context()
    if not ctx.is_training:
        # TODO Currently cannot build the wd_cost correctly at inference,
        # because ths vs_name used in inference can be '', therefore the
        # variable filter will fail
        return None

    # NOTE: this collection doesn't always grow with towers.
    # It is only added with variables that are newly created.
    if ctx.has_own_variables:   # be careful of the first tower (name='')
        losses = ctx.get_collection_in_tower(tf.GraphKeys.REGULARIZATION_LOSSES)
    else:
        losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if len(losses) > 0:
        logger.info("regularize_cost_from_collection() found {} tensors in REGULARIZATION_LOSSES.".format(len(losses)))
        reg_loss = tf.add_n(losses, name=name)
        return reg_loss
    else:
        return None


@layer_register(use_scope=None)
def Dropout(x, *args, **kwargs):
    """
    A wrapper around `tf.layers.Dropout`.

    Args:
        is_training (bool): If None, will use the current :class:`tensorpack.tfutils.TowerContext`
            to figure out.
        kwargs: same as in `tf.layers.Dropout`.
    """
    tfargs = parse_args(
        args=args, kwargs=kwargs,
        args_names=['rate'],
        name_mapping={
            'is_training': 'training',
        }
    )
    if len(args) > 0:
        logger.warn(
            "The first positional argument to tensorpack.Dropout is the probability to keep rather than to drop. "
            "This is different from the rate argument in tf.layers.Dropout due to historical reasons."
            "To mimic tf.layers.Dropout, use keyword argument 'rate' instead")
        rate = 1 - tfargs.pop('rate')
    elif 'keep_prob' in tfargs:
        assert 'rate' not in tfargs, "Cannot set both keep_prob and rate!"
        rate = 1 - tfargs.pop('keep_prob')
    elif rate not in tfargs:
        rate = 0.5

    if tfargs.get('training', None) is None:
        tfargs['training'] = get_current_tower_context().is_training

    return tf.layers.dropout(x, rate=rate, **tfargs)
