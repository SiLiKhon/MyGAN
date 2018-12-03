"""
GAN with Cramer metric
"""

from typing import Callable, Optional, Tuple, List, Union

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
            train_op_func: Callable[[tf.Tensor, tf.Tensor, List[tf.Variable], List[tf.Variable]], tf.Operation],
            gp_factor: Optional[Union[tf.Tensor, float]] = None,
            gp_mode: str = ''
        ) -> None:
        """
        Arguments:

        gp_factor -- term to multiply the gradient penalty by (optional).

        gp_mode -- implemented modes are:
                     - 'wgan_gp_one_sided'
                     - 'zero_data_only'

        See MyGAN.__init__ docstring for other arguments.
        """
        super().__init__(
                generator_func,
                discriminator_func,
                lambda gen, real, inter, w: self._losses_func(gen, real, inter, w,
                                                           gp_factor=gp_factor,
                                                           gp_mode=gp_mode),
                train_op_func
            )

    @staticmethod
    def _losses_func(
            disc_output_gen: tf.Tensor,
            disc_output_real: tf.Tensor,
            disc_output_int: tf.Tensor,
            weights: Optional[tf.Tensor] = None,
            name: Optional[str] = None,
            gp_factor: Optional[Union[tf.Tensor, float]] = None,
            gp_mode: str = '',
        ) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.name_scope(name, "Losses"):
            if weights is None:
                weights = tf.ones(shape=[tf.shape(disc_output_gen)[0]])

            gen1 , gen2  = tf.split(disc_output_gen , 2, axis=0)
            real1, real2 = tf.split(disc_output_real, 2, axis=0)
            int1 , _     = tf.split(disc_output_int , 2, axis=0)
            w1   , w2    = tf.split(weights         , 2, axis=0)

            with tf.name_scope("generator_loss"):
                gen_loss = (
                        tf.reduce_sum(tf.norm(real1 - gen1 , axis=1) * w1     , axis=0) / tf.reduce_sum(w1     , axis=0)
                      + tf.reduce_sum(tf.norm(real2 - gen2 , axis=1) * w2     , axis=0) / tf.reduce_sum(w2     , axis=0)
                      - tf.reduce_sum(tf.norm(gen1  - gen2 , axis=1) * w1 * w2, axis=0) / tf.reduce_sum(w1 * w2, axis=0)
                      - tf.reduce_sum(tf.norm(real1 - real2, axis=1) * w1 * w2, axis=0) / tf.reduce_sum(w1 * w2, axis=0)
                    )
                gen_loss = tf.identity(gen_loss, name='gen_loss')

            if gp_factor is not None:
                with tf.name_scope("gradient_penalty"):
                    if gp_mode == 'wgan_gp_one_sided':
                        critic_int = (
                                tf.norm(int1 - gen2 , axis=1)
                              - tf.norm(int1 - real2, axis=1)
                            )
        
                        gradients = tf.gradients(critic_int, [int1])[0]
                        slopes = tf.norm(tf.reshape(gradients, [tf.shape(gradients)[0], -1]), axis=1)
                        penalty = tf.reduce_mean(tf.square(tf.maximum(tf.abs(slopes) - 1, 0)), axis=0)
                        penalty = tf.identity(penalty, name='gradient_penalty')
                    elif gp_mode == 'zero_data_only':
                        critic_data = (
                                tf.norm(real1 - gen2 , axis=1)
                              - tf.norm(real1 - real2, axis=1)
                            )
                        gradients = tf.gradients(critic_data, [real1])[0]
                        slopes = tf.norm(tf.reshape(gradients, [tf.shape(gradients)[0], -1]), axis=1)
                        penalty = tf.reduce_sum(tf.square(slopes) * w1 * w2, axis=0) / tf.reduce_sum(w1 * w2, axis=0)
                        penalty = tf.identity(penalty, name='gradient_penalty')
                    else:
                        raise NotImplementedError(gp_mode)

            with tf.name_scope("discriminator_loss"):
                disc_loss = (
                        gp_factor * penalty - gen_loss
                        if gp_factor is not None else
                        -gen_loss
                        )
                disc_loss = tf.identity(disc_loss, name='disc_loss')

            return gen_loss, disc_loss

def cramer_gan() -> CramerGAN:
    """Build a CramerGAN with default architecture"""
    return CramerGAN(
        generator_func=deep_wide_generator,
        discriminator_func=deep_wide_discriminator,
        train_op_func=adversarial_train_op_func
    )
