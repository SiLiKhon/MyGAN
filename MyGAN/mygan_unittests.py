import unittest

import numpy as np
import tensorflow as tf

import MyGAN.dataset as mds
import MyGAN.mygan as mg

class UnitTestsMyGan(unittest.TestCase):
    def test_basic(self):
        N = 1000
        gan = mg.cramer_gan()
        ds = mds.Dataset(
            X=np.ones(shape=(N, 1), dtype=np.float32),
            Y=np.random.normal(loc=0.0, scale=2., size=(N, 1)).astype(np.float32)
        )

        gan.build_graph(ds, 100)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(gan._train_op)