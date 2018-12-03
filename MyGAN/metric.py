"""
TF metric functions
"""

from typing import Optional

import tensorflow as tf


def energy_distance_bootstrap(
        X: tf.Tensor,
        Y: tf.Tensor,
        w: Optional[tf.Tensor] = None,
        n_samples: int = 100
    ) -> tf.Tensor:
    with tf.control_dependencies([tf.assert_equal(tf.shape(X), tf.shape(Y))]):
        N = tf.shape(X)[0]
        mid = N // 2
        if w is None:
            w = tf.ones(shape=N, dtype=X.dtype)

        ids = tf.multinomial(tf.ones(shape=(n_samples, tf.shape(X)[0])), tf.shape(X)[0])

        X_batches = tf.gather(X, ids)
        Y_batches = tf.gather(Y, ids)
        w_batches = tf.gather(w, ids)

        X1, X2 = X_batches[:,:mid], X_batches[:,mid:]
        Y1, Y2 = Y_batches[:,:mid], Y_batches[:,mid:]
        w1, w2 = w_batches[:,:mid], w_batches[:,mid:]

        result = (
            tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(X1 - Y1), axis=2)) * w1     , axis=1) / tf.reduce_sum(w1     , axis=1)
          + tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(X2 - Y2), axis=2)) * w2     , axis=1) / tf.reduce_sum(w2     , axis=1)
          - tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(X2 - X1), axis=2)) * w1 * w2, axis=1) / tf.reduce_sum(w1 * w2, axis=1)
          - tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(Y2 - Y1), axis=2)) * w1 * w2, axis=1) / tf.reduce_sum(w1 * w2, axis=1)
        )
        return result
