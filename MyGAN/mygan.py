"""
GAN base class
"""

from typing import Callable, Tuple, List

import tensorflow as tf
import keras.layers as ll

Op_T2T = Callable[[tf.Tensor], tf.Tensor]

class MyGAN:
    """
    Base class for GANs
    """

    def __init__(
            self,
            generator: Op_T2T,
            discriminator: Op_T2T,
            losses_func: Callable[[Op_T2T, Op_T2T], Tuple[tf.Tensor, tf.Tensor]]
        ) -> None:
        """
        Constructor.
        losses_func is supposed to construct losses from generator and discriminator outputs:
            
            generator_loss, discriminator_loss = losses_func(generator(...), discriminator(...))

        """
        self.generator = generator
        self.discriminator = discriminator
        self.losses_func = losses_func


class CramerGAN(MyGAN):
    def __init__(
            self,
        ):
        pass


def get_dense(
        num_layers: int,
        neurons_per_layer: int = 128,
        activation: str = 'relu'
    ) -> List[ll.Dense]:
    """ Get a list of dense layers """
    return [ll.Dense(neurons_per_layer, activation=activation) for i in range(num_layers)]
