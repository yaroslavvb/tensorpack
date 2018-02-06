#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tflayer.py

import tensorflow as tf
import six

from ..tfutils.common import get_tf_version_number
from ..tfutils.varreplace import custom_getter_scope


def parse_args(args, kwargs, args_names, name_mapping):
    posarg_dic = {}
    assert len(args) <= len(args_names), \
        "Please use kwargs to call the model, except the following arguments: {}".format(','.join(args_names))
    for pos_arg, name in zip(args, args_names):
        posarg_dic[name] = pos_arg

    ret = {}
    for name, arg in six.iteritems(kwargs):
        newname = name_mapping.get(name, None)
        if newname is not None:
            assert newname not in kwargs, \
                "Argument {} and {} conflicts!".format(name, newname)
        else:
            newname = name
        ret[newname] = arg
    ret.update(posarg_dic)  # Let pos arg overwrite kw arg, for argscope
    return ret


def rename_get_variable(mapping):
    """
    Args:
        mapping(dict): an old -> new mapping for variable basename. e.g. {'kernel': 'W'}
    """
    def custom_getter(getter, name, *args, **kwargs):
        splits = name.split('/')
        basename = splits[-1]
        if basename in mapping:
            basename = mapping[basename]
            splits[-1] = basename
            name = '/'.join(splits)
        return getter(name, *args, **kwargs)
    return custom_getter_scope(custom_getter)


def monkeypatch_tf_layers():
    if get_tf_version_number() < 1.4:
        if not hasattr(tf.layers, 'Dense'):
            from tensorflow.python.layers.core import Dense
            tf.layers.Dense = Dense

            from tensorflow.python.layers.normalization import BatchNormalization
            tf.layers.BatchNormalization = BatchNormalization

            from tensorflow.python.layers.convolutional import Conv2DTranspose
            tf.layers.Conv2DTranspose = Conv2DTranspose


monkeypatch_tf_layers()
