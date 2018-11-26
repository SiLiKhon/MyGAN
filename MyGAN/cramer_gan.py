"""
GAN with Cramer metric
"""

from typing import Callable, Optional, Tuple, List

import tensorflow as tf

from .mygan import MyGAN
from .nns import deep_wide_generator, deep_wide_discriminator
from .train_utils import adversarial_train_op_func

class CramerGAN(MyGAN):
    """
    GAN with Cramer metric
    """
    def __init__(
            self,
            generator_func: Callable[[tf.Tensor, int], tf.Tensor],
            discriminator_func: Callable[[tf.Tensor], tf.Tensor],
            train_op_func: Callable[[tf.Tensor, tf.Tensor, List[tf.Variable], List[tf.Variable]], tf.Operation]
        ) -> None:
        super().__init__(
                generator_func,
                discriminator_func,
                self._losses_func,
                train_op_func
            )

    @staticmethod
    def _losses_func(
            disc_output_gen: tf.Tensor,
            disc_output_real: tf.Tensor,
            disc_output_int: tf.Tensor,
            weights: Optional[tf.Tensor] = None
        ) -> Tuple[tf.Tensor, tf.Tensor]:
        if weights is None:
            weights = tf.ones(shape=[tf.shape(disc_output_gen)[0]])
        gen1 , gen2  = tf.split(disc_output_gen , 2, axis=0)
        real1, real2 = tf.split(disc_output_real, 2, axis=0)
        int1 , _     = tf.split(disc_output_int , 2, axis=0)
        w1   , w2    = tf.split(weights         , 2, axis=0)

        gen_loss = (
                tf.reduce_sum(tf.norm(real1 - gen1 , axis=1) * w1     , axis=0) / tf.reduce_sum(w1     , axis=0)
              + tf.reduce_sum(tf.norm(real2 - gen2 , axis=1) * w2     , axis=0) / tf.reduce_sum(w2     , axis=0)
              - tf.reduce_sum(tf.norm(gen1  - gen2 , axis=1) * w1 * w2, axis=0) / tf.reduce_sum(w1 * w2, axis=0)
              - tf.reduce_sum(tf.norm(real1 - real2, axis=1) * w1 * w2, axis=0) / tf.reduce_sum(w1 * w2, axis=0)
            )
        
        critic_int = (
                tf.norm(int1 - gen2 , axis=1)
              - tf.norm(int1 - real2, axis=1)
            )
        
        gradients = tf.gradients(critic_int, [int1])[0]
        slopes = tf.norm(tf.reshape(gradients, [tf.shape(gradients)[0], -1]), axis=1)
        penalty = tf.reduce_sum(tf.square(tf.maximum(tf.abs(slopes) - 1, 0)), axis=0)

        disc_loss = -gen_loss + penalty

        return gen_loss, disc_loss

def cramer_gan() -> CramerGAN:
    """Build a CramerGAN with default architecture"""
    return CramerGAN(
        generator_func=deep_wide_generator,
        discriminator_func=deep_wide_discriminator,
        train_op_func=adversarial_train_op_func
    )
