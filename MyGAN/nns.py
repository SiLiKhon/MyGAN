"""
Basic NN architectures
"""

from typing import List, Optional, Callable, Union, Sequence

import tensorflow as tf

def get_dense(
        input: tf.Tensor,
        parameters: Sequence[Union[int, float]],
        activations: List[Optional[Callable[[tf.Tensor], tf.Tensor]]],
        kernel_initializer: tf.keras.initializers.Initializer = tf.initializers.variance_scaling(scale=2.),
        mode: Optional[tf.Tensor] = None,
        name: str = "dense"
    ) -> tf.Tensor:
    """
    Build a simple NN consisting of dense layers.

    Arguments:

    input -- input tensor.
    parameters -- list of ints (numbers of neurons for each layer) and floats (dropout probabilities).
    activations -- list of activation functions for each layer. Should contain None
        elements for each layer without an activation function.
    mode -- tensor evaluating to either 'train' or 'test' string (mandatory when using dropout layers).
    name -- name of the layers, optional (default = "dense").
    """
    assert len(parameters) == len(activations)
    
    output = input

    for i, (p, a) in enumerate(zip(parameters, activations)):
        if isinstance(p, int):
            output = tf.layers.dense(
                    output,
                    units=p,
                    activation=a,
                    kernel_initializer=kernel_initializer,
                    name="{}_{}".format(name, i)
                )
        elif isinstance(p, float):
            assert 0. < p < 1.
            assert mode is not None, "Dropout layer used, but mode was not provided"
            assert a is None, "Got an activation for a dropout layer"
            output = tf.layers.dropout(
                    output,
                    rate=p,
                    training=tf.equal(mode, 'train'),
                    name="{}_dropout_{}".format(name, i)
                )
        else:
            raise NotImplementedError("Parameter '{}' of type '{}'".format(p, type(p)))
    return output

def deep_wide_generator(
        input: tf.Tensor,
        n_out: int,
        n_latent: int = 32,
        depth: int = 7,
        width: int = 64,
        dropout_rates: Optional[List[Union[float, None]]] = None,
        mode: Optional[tf.Tensor] = None,
        activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu
    ) -> tf.Tensor:
    """
    Build a generator of a simple dense NN structure.

    Arguments:

    input -- input tensor.
    n_out -- size of the output (Y) space.
    n_latent -- latent space size, optional (default = 32).
    depth -- number of dense layers, optional (default = 7).
    width -- number of neurons per layer, optional (default = 64).
    dropout_rates -- list of dropout probabilities. Should be of length (depth - 1).
                     Place 'None' values where dropout is not wanted.
    mode -- tensor evaluating to either 'train' or 'test' string (mandatory when using dropout layers).
    activation -- activation function for all but the last layer (default = tf.nn.relu).
    """
    noise = tf.random_normal([tf.shape(input)[0], n_latent], dtype=input.dtype)
    input = tf.concat([noise, input], axis=1)
    activations = [activation for _ in range(depth - 1)] # type: List[Optional[Callable[[tf.Tensor], tf.Tensor]]]
    activations += [None]
    params = [width for _ in range(depth - 1)] # type: List[Union[float, int]]
    params += [n_out]
    if dropout_rates is not None:
        assert len(dropout_rates) == depth - 1
        for pos, rate in enumerate(dropout_rates[::-1], 1):
            if rate is not None:
                activations.insert(depth - pos, None)
                params     .insert(depth - pos, rate)

    return get_dense(input, params, activations, mode=mode)

def deep_wide_discriminator(
        input: tf.Tensor,
        depth: int = 7,
        width: int = 64,
        n_out: int = 128,
        dropout_rates: Optional[List[Union[float, None]]] = None,
        mode: Optional[tf.Tensor] = None,
        activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu
    ) -> tf.Tensor:
    """
    Build a discriminator of a simple dense NN structure.

    Arguments:

    input -- input tensor.
    depth -- number of dense layers, optional (default = 7).
    width -- number of neurons per layer, optional (default = 64).
    n_out -- size of the output space, optional (default = 128).
    dropout_rates -- list of dropout probabilities. Should be of length (depth - 1).
                     Place 'None' values where dropout is not wanted.
    mode -- tensor evaluating to either 'train' or 'test' string (mandatory when using dropout layers).
    activation -- activation function for all but the last layer (default = tf.nn.relu).
    """
    activations = [activation for _ in range(depth - 1)] # type: List[Optional[Callable[[tf.Tensor], tf.Tensor]]]
    activations += [None]
    params = [width for _ in range(depth - 1)] # type: List[Union[float, int]]
    params += [n_out]
    if dropout_rates is not None:
        assert len(dropout_rates) == depth - 1
        for pos, rate in enumerate(dropout_rates[::-1], 1):
            if rate is not None:
                activations.insert(depth - pos, None)
                params     .insert(depth - pos, rate)

    return get_dense(input, params, activations, mode=mode)

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
