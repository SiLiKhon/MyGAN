"""
GAN with Cramer metric
"""

from typing import Callable, Optional, Tuple, List, Union

import tensorflow as tf

from .mygan import MyGAN, TIn, TOut
from .nns import deep_wide_generator, deep_wide_discriminator
from .train_utils import adversarial_train_op_func

class CramerGAN(MyGAN):
    """
    GAN with Cramer metric
    """
    epsilon = 1e-9

    def __init__(
            self,
            generator_func: Callable[[TIn], TOut],
            discriminator_func: Callable[[TIn, TOut], tf.Tensor],
            train_op_func: Callable[[tf.Tensor, tf.Tensor, List[tf.Variable], List[tf.Variable]], tf.Operation],
            gp_factor: Optional[Union[tf.Tensor, float]] = None,
            gp_mode: str = '',
            surrogate: bool = False
        ) -> None:
        """
        Arguments:

        gp_factor -- term to multiply the gradient penalty by (optional).

        gp_mode -- implemented modes are:
                     - 'wgan_gp_one_sided'
                     - 'zero_data_only'

        See MyGAN.__init__ docstring for other arguments.
        """
        super().__init__( # type: ignore
                generator_func,
                discriminator_func,
                lambda gan: self._losses_func(gan,
                                              gp_factor=gp_factor,
                                              gp_mode=gp_mode,
                                              surrogate=surrogate),
                train_op_func
            )


    @staticmethod
    def _losses_func(
            gan: MyGAN,
            name: Optional[str] = None,
            gp_factor: Optional[Union[tf.Tensor, float]] = None,
            gp_mode: str = '',
            surrogate: bool = False
        ) -> Tuple[tf.Tensor, tf.Tensor]:
        eps = CramerGAN.epsilon

        weights = gan._W
        disc_output_gen  = gan._discriminator_output_gen
        disc_output_real = gan._discriminator_output_real
        disc_output_int  = gan._discriminator_output_int
        disc_input_X      = gan._X 
        disc_input_Y_real = gan._Y
        disc_input_Y_gen  = gan._generator_output
        disc_input_Y_int  = gan._Y_interpolates

        with tf.name_scope(name, "Losses"):
            if weights is None:
                weights = tf.ones(shape=[tf.shape(disc_output_gen)[0]])

            gen1 , gen2  = tf.split(disc_output_gen , 2, axis=0)
            real1, real2 = tf.split(disc_output_real, 2, axis=0)
            int1 , _     = tf.split(disc_output_int , 2, axis=0)
            w1   , w2    = tf.split(weights         , 2, axis=0)
            if surrogate:
                real2 = tf.zeros_like(real1)

            with tf.name_scope("generator_loss"):
                gen_loss = tf.reduce_sum(
                        w1 * w2 * (
                            tf.norm(real1 - gen2  + eps, axis=1)
                          + tf.norm(real2 - gen1  + eps, axis=1)
                          - tf.norm(gen1  - gen2  + eps, axis=1)
                          - tf.norm(real1 - real2 + eps, axis=1)
                        ),
                        axis=0
                    ) / tf.reduce_sum(w1 * w2, axis=0)
        
                gen_loss = tf.identity(gen_loss, name='gen_loss')

            if gp_factor is not None:
                with tf.name_scope("gradient_penalty"):
                    if gp_mode == 'wgan_gp_one_sided':
                        critic_int = (
                                tf.norm(int1 - gen2  + eps, axis=1)
                              - tf.norm(int1 - real2 + eps, axis=1)
                            )
        
                        grad1, grad2 = tf.gradients(critic_int, [disc_input_X, disc_input_Y_int])
                        gradients = tf.concat([
                                tf.reshape(grad1, [tf.shape(grad1)[0], -1]),
                                tf.reshape(grad2, [tf.shape(grad2)[0], -1]),
                            ], axis=1)
                        slopes = tf.norm(tf.reshape(gradients, [tf.shape(gradients)[0], -1]) + eps, axis=1)
                        penalty = tf.reduce_mean(tf.square(tf.maximum(tf.abs(slopes) - 1, 0)), axis=0)
                        penalty = tf.identity(penalty, name='gradient_penalty')
                    elif gp_mode == 'zero_data_only':
                        real11, real12 = tf.split(real1, 2, axis=0)
                        real21, real22 = tf.split(real2, 2, axis=0)
                        w11   , w12    = tf.split(w1   , 2, axis=0)
                        w21   , w22    = tf.split(w2   , 2, axis=0)
                        critic_data = (
                                tf.norm(real11 - real12 + eps, axis=1)
                              + tf.norm(real21 - real22 + eps, axis=1)
                              - tf.norm(real11 - real21 + eps, axis=1)
                              - tf.norm(real12 - real22 + eps, axis=1)
                            )
                        grad1, grad2 = tf.gradients(critic_data, [disc_input_X, disc_input_Y_real])
                        gradients = tf.concat([
                                tf.reshape(grad1, [tf.shape(grad1)[0], -1]),
                                tf.reshape(grad2, [tf.shape(grad2)[0], -1]),
                            ], axis=1)
                        slopes = tf.norm(tf.reshape(gradients, [tf.shape(gradients)[0], -1]) + eps, axis=1)
                        penalty = tf.reduce_sum(tf.square(slopes) * w11 * w12 * w21 * w22, axis=0) / tf.reduce_sum(w11 * w12 * w21 * w22, axis=0)
                        penalty = tf.identity(penalty, name='gradient_penalty')
                    else:
                        raise NotImplementedError(gp_mode)

            with tf.name_scope("discriminator_loss"):
                disc_loss = (
                        gp_factor * penalty - gen_loss
                        if gp_factor is not None else
                        tf.negative(gen_loss)
                        )
                disc_loss = tf.identity(disc_loss, name='disc_loss')

            return gen_loss, disc_loss

def cramer_gan(num_target_features: int) -> CramerGAN:
    """Build a CramerGAN with default architecture"""
    return CramerGAN( # type: ignore
        generator_func=lambda X: deep_wide_generator(X, num_target_features),
        discriminator_func=lambda X, Y: deep_wide_discriminator(
            tf.concat([X, Y], axis=1) # type: ignore
        ),
        train_op_func=adversarial_train_op_func
    )
