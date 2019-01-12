import numpy as np
import tensorflow as tf
import os
np.set_printoptions(formatter={'float_kind': lambda x: '%.2f' % x})


def print_vars(sess):
    print('print vars')
    for _var in tf.global_variables():
        assert _var.dtype.name == 'float32_ref', _var.name
        var = sess.run(_var)
        if _var in tf.trainable_variables():
            print('T', end='')
        else:
            print(' ', end='')
        if _var in tf.moving_average_variables():
            print('A', end='')
        else:
            print(' ', end='')
        if _var in tf.model_variables():
            print('M', end='')
        else:
            print(' ', end='')
        if _var in tf.local_variables():
            print('L', end='')
        else:
            print(' ', end='')
        print('', _var.name, var.shape, var.ravel()[0])


def w(shape):###
    return tf.get_variable('w', shape, initializer=tf.variance_scaling_initializer())###


###def w(shape):###
###    return tf.get_variable('w', shape, initializer=tf.initializers.random_normal(0.0, 0.02))###


def b(shape):
    return tf.get_variable('b', shape, initializer=tf.zeros_initializer())


def conv(mlp, c_out):
    c_in = mlp.get_shape()[3].value
    return tf.nn.conv2d(mlp, w([3, 3, c_in, c_out]), [1, 1, 1, 1], 'SAME')


def bias(mlp):
    c_in = mlp.get_shape()[-1].value
    return tf.nn.bias_add(mlp, b([c_in]))


def bn(mlp, is_training):
    """
        Batch-Normalization
    """
    ### return bias(mlp)
    return tf.contrib.layers.batch_norm(mlp, is_training=is_training, updates_collections=None)


def pool(mlp):
    """
        Conduct max-pooling operation
    """
    # 1，3，3，1？
    return tf.nn.max_pool(mlp, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
