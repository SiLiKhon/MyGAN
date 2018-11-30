"""
Useful functions for tensorboard monitoring.
"""

import os
import io
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import PIL

def make_histogram(
        summary_name: str,
        input: tf.Tensor,
        input_w: Optional[tf.Tensor] = None,
        reference: Optional[tf.Tensor] = None,
        reference_w: Optional[tf.Tensor] = None,
        bins: np.ndarray = 100,
        label: str = '',
        label_ref: str = '',
        alpha: float = 0.7,
        figsize: Tuple[float, float] = (7., 7.)
    ) -> tf.Tensor:
    if (label_ref is not None) or (reference_w is not None):
        assert reference is not None, "Reference label/weights given, but reference is None"
    
    def _array_to_hist_img(
            input: np.ndarray,
            input_w: Optional[np.ndarray] = None,
            reference: Optional[np.ndarray] = None,
            reference_w: Optional[np.ndarray] = None
        ) -> np.ndarray:
        _bins = bins

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if reference is not None:
            _, _bins, _ = ax.hist(reference, bins=_bins, label=label_ref, weights=reference_w)
        ax.hist(input, bins=_bins, label=label, alpha=alpha, weights=input_w)
        if label or label_ref:
            ax.legend()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        img = PIL.Image.open(buf)
        return np.array(img.getdata(), dtype=np.uint8).reshape(1, img.size[0], img.size[1], -1)

    if input_w is None:
        input_w = tf.ones(dtype=input.dtype, shape=tf.shape(input)[0])

    if reference is None:
        fig = tf.py_func(_array_to_hist_img, [input, input_w], tf.uint8)
    else:
        if reference_w is None:
            reference_w = tf.ones(dtype=reference.dtype, shape=tf.shape(reference)[0])
        fig = tf.py_func(_array_to_hist_img, [input, input_w, reference, reference_w], tf.uint8)
    return tf.summary.image(summary_name, fig)
