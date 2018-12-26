"""Weighted two-sample Kolmogorov-Smirnov statistic calculation with TensorFlow"""

from typing import Optional, Tuple

import tensorflow as tf

_T = tf.transpose

def get_ecdf(
        sample: tf.Tensor,
        weights: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Get empirical CDF from a weighted 1D sample
    """

    if weights is None:
        weights = tf.ones_like(sample)

    with tf.control_dependencies([tf.assert_equal(tf.shape(sample), tf.shape(weights))]):
        i = tf.contrib.framework.argsort(sample, axis=0)

        x = _T(tf.batch_gather(
                    _T(sample), _T(i)
                ))
        w = _T(tf.batch_gather(
                    _T(weights), _T(i)
                ))

        w_cumsum = tf.cumsum(w, axis=0)

        smallest_wsum = tf.reduce_min(w_cumsum[-1])
        with tf.control_dependencies([tf.assert_greater(smallest_wsum, tf.zeros_like(smallest_wsum))]):
            w_cumsum /= w_cumsum[-1]

    return x, w_cumsum


def _interleave_ecdfs(
                x1: tf.Tensor,
                y1: tf.Tensor,
                x2: tf.Tensor,
                y2: tf.Tensor
            ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Interleave two eCDFs by their argument
    """

    x = tf.contrib.framework.sort(tf.concat([x1, x2], axis=0), axis=0)
    y1 = tf.concat([tf.zeros(shape=(1, tf.shape(y1)[1]), dtype=y1.dtype), y1], axis=0)
    y2 = tf.concat([tf.zeros(shape=(1, tf.shape(y2)[1]), dtype=y2.dtype), y2], axis=0)
    return (
        x,
        _T(tf.batch_gather(_T(y1), tf.searchsorted(_T(x1), _T(x), side='right'))),
        _T(tf.batch_gather(_T(y2), tf.searchsorted(_T(x2), _T(x), side='right')))
    )

def _make_rank2(x: tf.Tensor):
    with tf.control_dependencies([tf.assert_rank_in(x, (1, 2))]):
        result = tf.reshape(x, [tf.shape(x)[0], -1])
    return result

def ks_2samp_w(
            data1: tf.Tensor,
            data2: tf.Tensor,
            w1: Optional[tf.Tensor] = None,
            w2: Optional[tf.Tensor] = None
        ) -> float:
    """
    Calculate KS statistic on two weighted samples.
    
    The samples can be either 1D or 2D.
    2D case is treated as if there were shape[1] samples of size shape[0]. Therefore,
    the samples must have the same length along axis=1.
    """
    data1_ = _make_rank2(data1)
    data2_ = _make_rank2(data2)
    if w1 is not None:
        w1 = _make_rank2(w1)
    if w2 is not None:
        w2 = _make_rank2(w2)

    cdf1 = get_ecdf(data1_, w1)
    cdf2 = get_ecdf(data2_, w2)
    _, cdf1_i, cdf2_i = _interleave_ecdfs(*cdf1, *cdf2)
    return tf.reduce_max(tf.abs(cdf2_i - cdf1_i), axis=0)