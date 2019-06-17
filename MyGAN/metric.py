"""
TF metric functions
"""

from typing import Optional

import tensorflow as tf

from . import weighted_ks

def energy_distance_bootstrap(
        X: tf.Tensor,
        Y: tf.Tensor,
        w: Optional[tf.Tensor] = None,
        n_samples: int = 100
    ) -> tf.Tensor:
    with tf.control_dependencies([tf.assert_equal(tf.shape(X), tf.shape(Y))]):
        N = tf.shape(X)[0]
        X = tf.reshape(X, [N, -1])
        Y = tf.reshape(Y, [N, -1])
        mid = N // 2
        if w is None:
            w = tf.ones(shape=N, dtype=X.dtype)

        ids = tf.random_uniform(shape=[n_samples, N], maxval=N, dtype=tf.int32)

        X_batches = tf.gather(X, ids)
        Y_batches = tf.gather(Y, ids)
        w_batches = tf.gather(w, ids)

        X1, X2 = X_batches[:,:mid], X_batches[:,mid:]
        Y1, Y2 = Y_batches[:,:mid], Y_batches[:,mid:]
        w1, w2 = w_batches[:,:mid], w_batches[:,mid:]

        result = (
            tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(X1 - Y2), axis=2)) * w1 * w2, axis=1)
          + tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(X2 - Y1), axis=2)) * w1 * w2, axis=1)
          - tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(X2 - X1), axis=2)) * w1 * w2, axis=1)
          - tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(Y2 - Y1), axis=2)) * w1 * w2, axis=1)
        ) / tf.reduce_sum(w1 * w2, axis=1)
        return result


def sliced_ks(
            X1: tf.Tensor,
            X2: tf.Tensor,
            w1: Optional[tf.Tensor] = None,
            w2: Optional[tf.Tensor] = None,
            n_samples: int = 100,
    ) -> tf.Tensor:

    random_slice_coordinates = tf.random_normal(
                shape=(tf.shape(X1)[1], 1, n_samples),
                dtype=X1.dtype
            )
    
    X1_slices = tf.nn.convolution(
                    X1[..., tf.newaxis],
                    random_slice_coordinates,
                    padding='VALID'
                )[:,0,:]
    
    X2_slices = tf.nn.convolution(
                    X2[..., tf.newaxis],
                    random_slice_coordinates,
                    padding='VALID'
                )[:,0,:]

    w1_tiled = tf.ones_like(X1_slices) \
                if w1 is None else \
                 tf.tile(w1[...,tf.newaxis], [1, n_samples])

    w2_tiled = tf.ones_like(X2_slices) \
                if w2 is None else \
                 tf.tile(w2[...,tf.newaxis], [1, n_samples])

    ks_for_random_slices = tf.reduce_max(
                                weighted_ks.tf.ks_2samp_w(
                                            X1_slices,
                                            X2_slices,
                                            w1_tiled, w2_tiled
                                )
                            )

    return ks_for_random_slices 


def sliced_ks_in_loops(
            X1: tf.Tensor,
            X2: tf.Tensor,
            w1: Optional[tf.Tensor] = None,
            w2: Optional[tf.Tensor] = None,
            n_samples: int = 10,
            n_loops: int = 10
    ) -> tf.Tensor:
    i = tf.constant(0)
    biggest_ks = tf.constant(0., dtype=X1.dtype)
    condition = lambda i, biggest_ks: tf.less(i, n_loops)
    body = lambda i, biggest_ks: (tf.add(i, 1), tf.maximum(sliced_ks(X1, X2, w1, w2, n_samples), biggest_ks))
    return tf.while_loop(condition, body, [i, biggest_ks])[1]
