import unittest
import tempfile
import os.path

import numpy as np
import tensorflow as tf

from . import dataset as mds
from .cramer_gan import cramer_gan

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
        gan = cramer_gan()
        ds = mds.Dataset(
            X=np.ones(shape=(N, 1), dtype=np.float32),
            Y=Y
        )

        gan.build_graph(ds, 10000)

        summary_path = os.path.join(self.temp_directory, "test_basic")
        print("Summary path is: {}".format(summary_path))

        summary_writer = tf.summary.FileWriter(
                            logdir=summary_path,
                            graph=tf.get_default_graph(),
                            max_queue=100,
                            flush_secs=1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(100):
                _, summary = sess.run([gan._train_op, gan.merged_summary])
                summary_writer.add_summary(summary, i)
                if i % 20 == 0:
                    print("step {}".format(i))