"""
GAN base class
"""

from typing import Callable, Tuple, List, Optional

import tensorflow as tf

import MyGAN.dataset as mds

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
    noise = tf.random.normal([tf.shape(input)[0], n_latent])
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