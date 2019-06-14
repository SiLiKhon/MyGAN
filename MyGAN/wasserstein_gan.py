"""
GAN with Wasserstein metric
"""

from typing import Callable, Optional, Tuple, List, Union

import tensorflow as tf

from .mygan import MyGAN, TIn, TOut
from .nns import deep_wide_generator, deep_wide_discriminator
from .train_utils import adversarial_train_op_func

class WassersteinGAN(MyGAN):
    """
    GAN with Wasserstein metric
    """

    def __init__(
            self,
            generator_func: Callable[[TIn], TOut],
            discriminator_func: Callable[[TIn, TOut], tf.Tensor],
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
        super().__init__( # type: ignore
                generator_func,
                discriminator_func,
                lambda gan: self._losses_func(gan,
                                              gp_factor=gp_factor,
                                              gp_mode=gp_mode),
                train_op_func
            )


    @staticmethod
    def _losses_func(
            gan: MyGAN,
            name: Optional[str] = None,
            gp_factor: Optional[Union[tf.Tensor, float]] = None,
            gp_mode: str = ''
        ) -> Tuple[tf.Tensor, tf.Tensor]:
        weights = gan._W
        disc_output_gen  = tf.squeeze(gan._discriminator_output_gen )
        disc_output_real = tf.squeeze(gan._discriminator_output_real)
        disc_output_int  = tf.squeeze(gan._discriminator_output_int )
        disc_input_X      = gan._X 
        disc_input_Y_real = gan._Y
        disc_input_Y_int  = gan._Y_interpolates
        # Not used:
        # disc_input_Y_gen = gan._generator_output

        with tf.name_scope(name, "Losses"):
            if weights is None:
                weights = tf.ones(shape=[tf.shape(disc_output_gen)[0]])

            with tf.name_scope("generator_loss"):
                with tf.control_dependencies([
                        tf.assert_rank(disc_output_gen, 1)
                    ]):
                    gen_loss = tf.reduce_sum(
                            weights * (
                               disc_output_gen - disc_output_real 
                            ),
                            axis=0
                        ) / tf.reduce_sum(weights, axis=0)
            
                    gen_loss = tf.identity(gen_loss, name='gen_loss')

            if gp_factor is not None:
                with tf.name_scope("gradient_penalty"):
                    if gp_mode == 'wgan_gp_one_sided':
                        grad1, grad2 = tf.gradients(disc_output_int, [disc_input_X, disc_input_Y_int])
                        gradients = tf.concat([
                                tf.reshape(grad1, [tf.shape(grad1)[0], -1]),
                                tf.reshape(grad2, [tf.shape(grad2)[0], -1]),
                            ], axis=1)
                        slopes = tf.norm(tf.reshape(gradients, [tf.shape(gradients)[0], -1]), axis=1)
                        penalty = tf.reduce_mean(tf.square(tf.maximum(tf.abs(slopes) - 1, 0)), axis=0)
                        penalty = tf.identity(penalty, name='gradient_penalty')
                    elif gp_mode == 'zero_data_only':
                        grad1, grad2 = tf.gradients(disc_output_real, [disc_input_X, disc_input_Y_real])
                        gradients = tf.concat([
                                tf.reshape(grad1, [tf.shape(grad1)[0], -1]),
                                tf.reshape(grad2, [tf.shape(grad2)[0], -1]),
                            ], axis=1)
                        slopes = tf.norm(tf.reshape(gradients, [tf.shape(gradients)[0], -1]), axis=1)
                        penalty = tf.reduce_sum(tf.square(slopes) * weights, axis=0) / tf.reduce_sum(weights, axis=0)
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

