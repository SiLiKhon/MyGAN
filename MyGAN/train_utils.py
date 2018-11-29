"""
Some helper functions for training
"""

from typing import Callable, List

import tensorflow as tf

def _op_repeat_n(
        op_f: Callable[[], tf.Operation],
        n: int
    ) -> tf.Operation:
    assert n > 0

    op = op_f()
    n -= 1

    for _ in range(n):
        with tf.control_dependencies([op]):
            op = op_f()
    return op


def adversarial_train_op_func(
        generator_loss: tf.Tensor,
        discriminator_loss: tf.Tensor,
        generator_weights: List[tf.Variable],
        discriminator_weights: List[tf.Variable],
        n_gen_steps: int = 1,
        n_disc_steps: int = 5,
        optimizer: tf.train.Optimizer = tf.train.RMSPropOptimizer(0.0005)
    ) -> tf.Operation:
    """
    Build the adversarial train operation (n_disc_steps discriminator optimization steps
    followed by n_gen_steps generator optimization steps).

    Arguments:

    generator_loss -- generator loss.
    discriminator_loss -- discriminator loss.
    generator_weights -- list of generator trainable weights.
    discriminator_weights -- list of discriminator trainable weights.
    n_gen_steps -- number of generator update steps per single train operation,
        optional (default = 1).
    n_disc_steps -- number of discriminator update steps per single train
        operation, optional (default = 10).
    optimizer -- optimizer to use, optional (default = tf.train.RMSPropOptimizer(0.001))
    """
    disc_train_op = _op_repeat_n(
            lambda: optimizer.minimize(discriminator_loss, var_list=discriminator_weights),
            n_disc_steps
        )
    
    with tf.control_dependencies([disc_train_op]):
        gen_train_op = _op_repeat_n(
                lambda: optimizer.minimize(generator_loss, var_list=generator_weights),
                n_gen_steps
            )
    
    return gen_train_op

def create_mode(
        name: str = 'mode',
        name_scope: str = 'Mode/'
    ) -> tf.Tensor:
    with tf.name_scope(name_scope):
        mode = tf.placeholder_with_default('train', [], name=name)
    return mode
