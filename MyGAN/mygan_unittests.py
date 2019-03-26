import unittest
import tempfile
import os.path

import numpy as np
import tensorflow as tf

from . import dataset as mds
from .cramer_gan import cramer_gan
from . import tf_monitoring as tfmon
from .train_utils import create_mode

class UnitTestsMyGan(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.gettempdir()

    def test_basic(self):
        tf.reset_default_graph()


        N = 100000
        Y01 = np.random.normal(loc=0.0, scale=1., size=(N, 2)).astype(np.float32)
        Y2  = Y01.sum(axis=1).reshape((-1, 1)) / 2 + \
                np.random.normal(loc=0.0, scale=0.01, size=(N, 1)).astype(np.float32)
        Y = np.concatenate([Y01, Y2], axis=1)
        gan = cramer_gan(num_target_features=Y.shape[1])
        ds = mds.Dataset(
            X=np.ones(shape=(N, 1), dtype=np.float32),
            Y=Y
        )
        ds_train, ds_test = ds.split(test_size=0.2)
        print((len(ds_train), len(ds_test)))

        mode = create_mode()
        X_train, Y_train = ds_train.get_tf(
            batch_size=10000, cols=['X', 'Y']
        )
        X_test, Y_test = ds_test.get_tf(
            batch_size=len(ds_test), cols=['X', 'Y'], make_tf_ds=False
        )
        gan.build_graph(
            X_train=X_train, Y_train=Y_train,
            X_test=X_test, Y_test=Y_test, mode=mode
        )
        gan.make_summary_histogram('Y0', lambda Y: Y[:,0])
        gan.make_summary_energy(name='energy_distance_full')

        train_summary, val_summary = gan.get_merged_summaries()

        summary_path = os.path.join(self.temp_directory, "test_basic")
        print("Summary path is: {}".format(summary_path))
        summary_path_train = os.path.join(summary_path, 'train')
        summary_path_test  = os.path.join(summary_path, 'test' )

        summary_writer_train = tf.summary.FileWriter(
                                                logdir=summary_path_train,
                                                graph=tf.get_default_graph(),
                                                max_queue=100,
                                                flush_secs=1
                                            )
        summary_writer_test = tf.summary.FileWriter(
                                                logdir=summary_path_test,
                                                max_queue=100,
                                                flush_secs=1
                                            )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(10):
                _, summary = sess.run([gan.train_op, train_summary])
                summary_writer_train.add_summary(summary, i)
                if i % 5 == 0:
                    summary = sess.run(val_summary, {gan.mode : 'test'})
                    summary_writer_test.add_summary(summary, i)
                    print("step {}".format(i))