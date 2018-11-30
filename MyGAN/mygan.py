"""
GAN base class
"""

from typing import Callable, Tuple, List, Optional

import tensorflow as tf

from . import dataset as mds
from . import tf_monitoring as tfmon

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

        self.summary_histograms: List[tf.Tensor] = []

    def build_graph(
            self,
            train_ds: mds.Dataset,
            test_ds: mds.Dataset,
            batch_size: int,
            mode: tf.Tensor,
            seed: Optional[int] = None,
            noise_std: Optional[float] = None
        ) -> None:
        """
        Build the graph.

        Arguments:

        train_ds -- MyGAN.dataset.Dataset object with data to train on.
        test_ds -- MyGAN.dataset.Dataset object with data to test on.
        batch_size -- batch size.
        mode -- tensor evaluating to either 'train' or 'test' string.
        seed -- random seed to be used for dataset shuffling, optional.
        noise_std -- standard deviation of noise to be added to data (both X and Y), optional.
        """

        assert train_ds.check_similar(test_ds)

        self.mode = mode

        self._train_ds = train_ds
        self._test_ds = test_ds
        self._weighted = (train_ds.W is not None)
        cols = ['X', 'Y', 'XY']
        if self._weighted:
            cols += 'W'

        with tf.name_scope('Inputs'):
            with tf.name_scope('Train'):
                tf_inputs = train_ds.get_tf(
                                batch_size=batch_size,
                                cols=cols,
                                seed=seed,
                                noise_std=noise_std
                            )

            with tf.name_scope('Test'):
                tf_inputs_test = test_ds.get_tf(
                                     batch_size=len(test_ds),
                                     cols=cols,
                                     make_tf_ds=False
                                 )


            self._X, self._Y, self._XY = tf.case(
                    {
                        tf.equal(self.mode, 'train') : lambda: tf_inputs     [:3],
                        tf.equal(self.mode, 'test' ) : lambda: tf_inputs_test[:3]
                    },
                    exclusive=True
                )
            
            self._W = None
            if self._weighted:
                self._W = tf.case(
                        {
                            tf.equal(self.mode, 'train') : lambda: tf_inputs     [3],
                            tf.equal(self.mode, 'test' ) : lambda: tf_inputs_test[3]
                        },
                        exclusive=True
                    )

        with tf.variable_scope(self.gen_scope):
            self._generator_output = self.generator_func(self._X, train_ds.ny)

        with tf.variable_scope(self.disc_scope):
            self._discriminator_output_real = self.discriminator_func(self._XY)

        with tf.variable_scope(self.disc_scope, reuse=True):
            self._discriminator_output_gen  = self.discriminator_func(
                    tf.concat([self._X, self._generator_output], axis=1)
                )

        with tf.variable_scope(self.disc_scope, reuse=True):
            alpha = tf.random_uniform(shape=[tf.shape(self._XY)[0], 1], minval=0., maxval=1.)
            interpolates = (
                    alpha * self._XY
                  + (1 - alpha) * tf.concat([self._X, self._generator_output], axis=1)
                )
            self._discriminator_output_int = self.discriminator_func(interpolates)

        with tf.name_scope("Training"):
            self._gen_loss, self._disc_loss = self.losses_func(
                    self._discriminator_output_gen,
                    self._discriminator_output_real,
                    self._discriminator_output_int,
                    self._W
                )

            self.train_op = self.train_op_func(
                    self._gen_loss,
                    self._disc_loss,
                    self.get_gen_weights(),
                    self.get_disc_weights()
                )
        
        with tf.control_dependencies([self.train_op]):
            gen_loss_summary  = tf.summary.scalar('Generator_loss'    , self._gen_loss )
            disc_loss_summary = tf.summary.scalar('Discriminator_loss', self._disc_loss)
            self.merged_summary = tf.summary.merge([gen_loss_summary, disc_loss_summary])
    
    def get_gen_weights(self) -> List[tf.Variable]:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.gen_scope)
    
    def get_disc_weights(self) -> List[tf.Variable]:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.disc_scope)

    def make_summary_histogram(
            self,
            name: str,
            func: Callable[[tf.Tensor], tf.Tensor],
            name_scope: str = 'Monitoring/'
        ) -> None:
        with tf.name_scope(name_scope):
            self.summary_histograms.append(
                            tfmon.make_histogram(
                                            summary_name=name,
                                            input=func(self._generator_output),
                                            input_w=self._W,
                                            reference=func(self._Y),
                                            reference_w=self._W,
                                            label='Generated',
                                            label_ref='Real'
                                        )
                        )