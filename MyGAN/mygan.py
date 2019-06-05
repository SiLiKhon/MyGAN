"""
GAN base class
"""

from typing import Callable, Tuple, List, Optional, TypeVar, Generic
import contextlib

import tensorflow as tf

from . import tf_monitoring as tfmon
from .metric import energy_distance_bootstrap, sliced_ks_in_loops

TIn  = TypeVar("TIn")
TOut = TypeVar("TOut")
T_GAN = TypeVar('T_GAN', bound='MyGAN')

class MyGAN(Generic[TIn, TOut]):
    """
    Base class for GANs
    """

    def __init__(
            self,
            generator_func: Callable[[TIn], TOut],
            discriminator_func: Callable[[TIn, TOut], tf.Tensor],
            losses_func: Callable[[T_GAN], Tuple[tf.Tensor, tf.Tensor]],
            train_op_func: Callable[[tf.Tensor, tf.Tensor, List[tf.Variable], List[tf.Variable]], tf.Operation]
        ) -> None:
        """
        Constructor.

        Arguments:

        generator_func -- function to build the generator. Should follow the signature:
            generator_func(X) -> Y_gen. All the tf variables should follow the
            tf.get_variable paradigm to allow for weights reuse.
        discriminator_func -- function to build the discriminator. Should follow the
            signature: discriminator_func(X, Y) -> output_tensor. All the tf variables
            should follow the tf.get_variable paradigm to allow for weights reuse.
        losses_func -- function to construct losses:
                            losses_func(gan) -> (generator_loss, discriminator_loss)
                       the function may use the following variables of the gan object:
                         - gan._X (conditional variables)
                         - gan._Y (target variables)
                         - gan._generator_output (generated Y)
                         - gan._Y_interpolates (interpolated between real and generated Y variables)
                         - gan._W (weights)
                         - gan._discriminator_output_gen (discriminator output on generated variables)
                         - gan._discriminator_output_real (discriminator output on real variables)
                         - gan._discriminator_output_int (discriminator output on interpolated variables)
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

        self.train_summaries: List[tf.Tensor] = []
        self.test_summaries: List[tf.Tensor] = []

        self.summaries_finalized = False
        self._has_unclosed_train_hist_summary = False
        self._has_unclosed_test_hist_summary = False

    def build_graph(
            self,
            X_train: TIn,
            Y_train: TOut,
            X_test: TIn,
            Y_test: TOut,
            mode: tf.Tensor,
            W_train: Optional[tf.Tensor] = None,
            W_test: Optional[tf.Tensor] = None,
        ) -> None:
        """
        Build the graph.

        Arguments:

        X_train -- input to train on.
        Y_train -- output (target) to train on.
        X_test -- input to test on.
        Y_test -- output (target) to test on.
        mode -- tensor evaluating to either 'train' or 'test' string.
        W_train -- tensor with train sample weights (optional)
        W_test -- tensor with test sample weights (optional)
        """

        self.mode = mode

        assert (W_train is None) == (W_test is None), \
            "Both W_train and W_test should be either provided or omitted"

        self._weighted = (W_train is not None)

        with tf.name_scope('Inputs'):
            self._X, self._Y = tf.case(
                    {
                        tf.equal(self.mode, 'train') : lambda: [X_train, Y_train],
                        tf.equal(self.mode, 'test' ) : lambda: [X_test , Y_test ],
                    },
                    exclusive=True
                )
            self._XY = (self._X, self._Y)
            
            self._W = None
            if self._weighted:
                self._W = tf.case(
                        {
                            tf.equal(self.mode, 'train') : lambda: W_train,
                            tf.equal(self.mode, 'test' ) : lambda: W_test ,
                        },
                        exclusive=True
                    )

        with tf.variable_scope(self.gen_scope):
            self._generator_output = self.generator_func(self._X)
            self._generator_output_XY = (self._X, self._generator_output)

        with tf.variable_scope(self.disc_scope):
            self._discriminator_output_real = self.discriminator_func(*self._XY)

        with tf.variable_scope(self.disc_scope, reuse=True):
            self._discriminator_output_gen  = self.discriminator_func(
                    *self._generator_output_XY
                )

        with tf.variable_scope(self.disc_scope, reuse=True):
            alpha = tf.random_uniform(shape=[tf.shape(self._X)[0], 1], minval=0., maxval=1., dtype=self._X.dtype)
            self._Y_interpolates = (
                    alpha * self._Y
                  + (1 - alpha) * self._generator_output
                )
            self._discriminator_output_int = self.discriminator_func(self._X, self._Y_interpolates)

        with tf.name_scope("Training"):
            self._gen_loss, self._disc_loss = self.losses_func(self) # type: ignore

            self.train_op = self.train_op_func(
                    self._gen_loss,
                    self._disc_loss,
                    self.get_gen_weights(),
                    self.get_disc_weights()
                )
        
        with tf.control_dependencies([self.train_op]):
            gen_loss_summary  = tf.summary.scalar('Generator_loss'    , self._gen_loss )
            disc_loss_summary = tf.summary.scalar('Discriminator_loss', self._disc_loss)
            self.train_summaries += [gen_loss_summary, disc_loss_summary]
            self.test_summaries  += [gen_loss_summary, disc_loss_summary]
    
    def get_gen_weights(self) -> List[tf.Variable]:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.gen_scope)
    
    def get_disc_weights(self) -> List[tf.Variable]:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.disc_scope)

    def get_merged_summaries(self) -> Tuple[tf.Tensor, tf.Tensor]:
        if not self.summaries_finalized:
            with contextlib.ExitStack() as stack:
                if self._has_unclosed_train_hist_summary:
                    stack.enter_context(tf.control_dependencies(self.train_summaries))
                    close_all_figs_train_op = tfmon.close_all_figures_op()
                    stack.enter_context(tf.control_dependencies([close_all_figs_train_op]))
                self.train_summary = tf.summary.merge(self.train_summaries)
            with contextlib.ExitStack() as stack:
                if self._has_unclosed_test_hist_summary:
                    stack.enter_context(tf.control_dependencies(self.test_summaries))
                    close_all_figs_test_op = tfmon.close_all_figures_op()
                    stack.enter_context(tf.control_dependencies([close_all_figs_test_op]))
                self.test_summary = tf.summary.merge(self.test_summaries)
            self.summaries_finalized = True

        return self.train_summary, self.test_summary

    def make_summary_histogram(
            self,
            name: str,
            func: Callable[[TOut], tf.Tensor],
            name_scope: str = 'Monitoring/',
            train_summary: bool = False,
            test_summary: bool = True,
            autoclose_figure: bool = False 
        ) -> None:
        assert not self.summaries_finalized
        assert train_summary or test_summary

        with tf.name_scope(name_scope):
            hist_summary = tfmon.make_histogram(
                                        summary_name=name,
                                        input=func(self._generator_output),
                                        input_w=self._W,
                                        reference=func(self._Y),
                                        reference_w=self._W,
                                        label='Generated',
                                        label_ref='Real',
                                        close_fig=autoclose_figure
                                    )
            if train_summary:
                self.train_summaries.append(hist_summary)
                if not autoclose_figure:
                    self._has_unclosed_train_hist_summary = True
            if test_summary:
                self.test_summaries.append(hist_summary)
                if not autoclose_figure:
                    self._has_unclosed_test_hist_summary = True


    def make_summary_energy(
            self,
            name: str,
            projection_func: Callable[[TIn, TOut], tf.Tensor] = lambda X, Y: tf.concat([X, Y], axis=1),
            name_scope: str = 'Monitoring/',
            n_samples: int = 100,
            train_summary: bool = False,
            test_summary: bool = True
        ) -> None:
        assert not self.summaries_finalized
        assert train_summary or test_summary

        with tf.name_scope(name_scope):
            test, ref = (
                    projection_func(*self._generator_output_XY),
                    projection_func(*self._XY)
                )

            energy = energy_distance_bootstrap(test, ref, self._W, n_samples)
            summary_energy = tf.summary.histogram(name, energy)

            if train_summary:
                self.train_summaries.append(summary_energy)
            if test_summary:
                self.test_summaries.append(summary_energy)

    def make_summary_sliced_looped_ks(
            self,
            name: str,
            projection_func: Callable[[TIn, TOut], tf.Tensor] = lambda X, Y: tf.concat([X, Y], axis=1),
            name_scope: str = 'Monitoring/',
            n_samples: int = 10,
            n_loops: int = 10,
            train_summary: bool = True,
            test_summary: bool = True
        ) -> None:
        assert not self.summaries_finalized
        assert train_summary or test_summary

        with tf.name_scope(name_scope):
            ks_statistic = sliced_ks_in_loops(
                    projection_func(*self._generator_output_XY),
                    projection_func(*self._XY),
                    self._W,
                    self._W,
                    n_samples=n_samples,
                    n_loops=n_loops
                )
            summary_ks = tf.summary.scalar(name, ks_statistic)

            if train_summary:
                self.train_summaries.append(summary_ks)
            if test_summary:
                self.test_summaries.append(summary_ks)
