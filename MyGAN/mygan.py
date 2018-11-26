"""
GAN base class
"""

from typing import Callable, Tuple, List, Optional

import tensorflow as tf

from . import dataset as mds

class MyGAN:
    """
    Base class for GANs
    """

    def __init__(
            self,
            generator_func: Callable[[tf.Tensor, int], tf.Tensor],
            discriminator_func: Callable[[tf.Tensor], tf.Tensor],
            losses_func: Callable[[tf.Tensor, tf.Tensor, tf.Tensor, Optional[tf.Tensor]], Tuple[tf.Tensor, tf.Tensor]],
            train_op_func: Callable[[tf.Tensor, tf.Tensor, List[tf.Variable], List[tf.Variable]], tf.Operation]
        ) -> None:
        """
        Constructor.

        Arguments:

        generator_func -- function to build the generator. Should follow the signature:
            generator_func(input_X_tensor, num_output_features) -> output_Y_gen_tensor.
            All the tf variables should follow the tf.get_variable paradigm to allow for
            weights reuse.
        discriminator_func -- function to build the discriminator. Should follow the
            signature: discriminator_func(input_XY_tensor) -> output_tensor. All the tf
            variables should follow the tf.get_variable paradigm to allow for weights
            reuse.
        losses_func -- function to construct losses from discriminator outputs on generated
            samples, real samples and their linear interpolates (for gradient penalty):
            losses_func(discriminator(generated_samples), discriminator(real_samples),
            discriminator(interpolates), weights) -> (generator_loss, discriminator_loss)
        train_op_func -- function to build the training operation:
            train_op_func(generator_loss, discriminator_loss, generator_weights,
            discriminator_weights) -> training_operation.
        """

        self.generator_func = generator_func
        self.discriminator_func = discriminator_func
        self.losses_func = losses_func
        self.train_op_func = train_op_func

        self.gen_scope = 'Generator'
        self.disc_scope = 'Discriminator'


    def build_graph(
            self,
            train_ds: mds.Dataset,
            batch_size: int,
            seed: Optional[int] = None
        ) -> None:
        """
        Build the graph.

        Arguments:

        train_ds -- MyGAN.dataset.Dataset object with data to train on.
        batch_size -- batch size.
        seed -- random seed to be used for dataset shuffling.        
        """

        self._train_ds = train_ds
        self._weighted = (train_ds.W is not None)
        cols = ['X', 'Y', 'XY']
        if self._weighted:
            cols += 'W'

        tf_inputs = train_ds.get_tf(
                        batch_size=batch_size,
                        cols=cols,
                        seed=seed
                    )

        self._train_X, self._train_Y, self._train_XY = tf_inputs[:3]
        self._train_W = None
        if self._weighted:
            self._train_W = tf_inputs[3]

        with tf.variable_scope(self.gen_scope):
            self._generator_output = self.generator_func(self._train_X, train_ds.ny)

        with tf.variable_scope(self.disc_scope):
            self._discriminator_output_real = self.discriminator_func(self._train_XY)

        with tf.variable_scope(self.disc_scope, reuse=True):
            self._discriminator_output_gen  = self.discriminator_func(
                    tf.concat([self._train_X, self._generator_output], axis=1)
                )

        with tf.variable_scope(self.disc_scope, reuse=True):
            alpha = tf.random_uniform(shape=[batch_size, 1], minval=0., maxval=1.)
            interpolates = (
                    alpha * self._train_XY
                  + (1 - alpha) * tf.concat([self._train_X, self._generator_output], axis=1)
                )
            self._discriminator_output_int = self.discriminator_func(interpolates)

        with tf.name_scope("Training"):
            self._gen_loss, self._disc_loss = self.losses_func(
                    self._discriminator_output_gen,
                    self._discriminator_output_real,
                    self._discriminator_output_int,
                    self._train_W
                )
            
            self._train_op = self.train_op_func(
                    self._gen_loss,
                    self._disc_loss,
                    self.get_gen_weights(),
                    self.get_disc_weights()
                )
        
        with tf.control_dependencies([self._train_op]):
            gen_loss_summary  = tf.summary.scalar('Generator_loss'    , self._gen_loss )
            disc_loss_summary = tf.summary.scalar('Discriminator_loss', self._disc_loss)
            self.merged_summary = tf.summary.merge([gen_loss_summary, disc_loss_summary])
    
    def get_gen_weights(self) -> List[tf.Variable]:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.gen_scope)
    
    def get_disc_weights(self) -> List[tf.Variable]:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.disc_scope)
