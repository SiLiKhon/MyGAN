"""
Basic NN architectures
"""

from typing import List, Optional, Callable, Union

import tensorflow as tf

def get_dense(
        input: tf.Tensor,
        widths: List[int],
        activations: List[Optional[Callable[[tf.Tensor], tf.Tensor]]],
        name: str = "dense"
    ) -> tf.Tensor:
    """
    Build a simple NN consisting of dense layers.

    Arguments:

    input -- input tensor.
    widths -- list of numbers of neurons for each layer.
    activations -- list of activation functions for each layer. Should contain None
        elements for each layer without an activation function.
    name -- name of the layers, optional (default = "dense").
    """
    assert len(widths) == len(activations)
    
    output = input

    for i, (w, a) in enumerate(zip(widths, activations)):
        output = tf.layers.dense(
                output,
                units=w,
                activation=a,
                name="{}_{}".format(name, i)
            )
    return output

def deep_wide_generator(
        input: tf.Tensor,
        n_out: int,
        n_latent: int = 32,
        depth: int = 7,
        width: int = 64
    ) -> tf.Tensor:
    """
    Build a generator of a simple dense NN structure.

    Arguments:

    input -- input tensor.
    n_out -- size of the output (Y) space.
    n_latent -- latent space size, optional (default = 32).
    depth -- number of dense layers, optional (default = 7).
    width -- number of neurons per layer, optional (default = 64).
    """
    noise = tf.random_normal([tf.shape(input)[0], n_latent], dtype=input.dtype)
    input = tf.concat([noise, input], axis=1)
    return get_dense(
            input,
            [width      for _ in range(depth - 1)] + [n_out],
            [tf.nn.relu for _ in range(depth - 1)] + [None]
        )

def deep_wide_discriminator(
        input: tf.Tensor,
        depth: int = 7,
        width: int = 64,
        n_out: int = 128
    ) -> tf.Tensor:
    """
    Build a discriminator of a simple dense NN structure.

    Arguments:

    input -- input tensor.
    depth -- number of dense layers, optional (default = 7).
    width -- number of neurons per layer, optional (default = 64).
    n_out -- size of the output space, optional (default = 128).
    """
    return get_dense(
            input,
            [width      for _ in range(depth - 1)] + [n_out],
            [tf.nn.relu for _ in range(depth - 1)] + [None]
        )

def noise_layer(
        input: tf.Tensor,
        stddev: Union[float, tf.Tensor],
        mode: tf.Tensor
    ) -> tf.Tensor:
    """Simple noise layer"""
    noise = tf.case(
            {
                tf.equal(mode, 'train') : lambda: tf.random_normal(
                                                        shape=tf.shape(input),
                                                        dtype=input.dtype,
                                                        stddev=stddev
                                                    ),
                tf.equal(mode, 'test' ) : lambda: tf.zeros(
                                                        shape=tf.shape(input),
                                                        dtype=input.dtype
                                                    ),
            },
            exclusive=True
        )
    return input + noise
