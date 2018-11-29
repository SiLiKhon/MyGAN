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
        reference: Optional[tf.Tensor] = None,
        bins: np.ndarray = 100,
        label: str = '',
        label_ref: str = '',
        alpha: float = 0.7,
        figsize: Tuple[float, float] = (7., 7.)
    ) -> tf.Tensor:
    if label_ref:
        assert reference is not None, "Reference label given, but reference is None"
    
    def _array_to_hist_img(
            input: np.ndarray,
            reference: Optional[np.ndarray] = reference
        ) -> np.ndarray:
        _bins = bins

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if reference is not None:
            _, _bins, _ = ax.hist(reference, bins=_bins, label=label_ref)
        ax.hist(input, bins=_bins, label=label, alpha=alpha)
        if label or label_ref:
            ax.legend()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        img = PIL.Image.open(buf)
        return np.array(img.getdata(), dtype=np.uint8).reshape(1, img.size[0], img.size[1], -1)
    
    if reference is None:
        fig = tf.py_func(_array_to_hist_img, [input], tf.uint8)
    else:
        fig = tf.py_func(_array_to_hist_img, [input, reference], tf.uint8)
    return tf.summary.image(summary_name, fig)
