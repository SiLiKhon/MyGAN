import unittest
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import MyGAN.dataset as mds

class UnitTestsDataset(unittest.TestCase):
    """
    Basic tests for the Dataset class
    """

    TEST_SIZE: int = 10
    TMP_FILENAME: str = 'dataset_unittests_tmp_file.csv'

    def setUp(self):
        """
        Initialize some pandas DataFrame objects, create a csv file.
        """
        cols = ['x1', 'x2', 'x3', 'x4', 'x5',
                'y1', 'y2', 'w']
        self.df1 = pd.DataFrame({i : np.arange(1, self.TEST_SIZE + 1).astype(float)
                        for i in cols})
        for i, col in enumerate(cols):
            self.df1[col] /= (10**i)
        
        csv = \
r"""b,a,c,d
1,2,3,4
5,6,7,8
9,0,1,2
"""
        
        assert not os.path.isfile(self.TMP_FILENAME)

        with open(self.TMP_FILENAME, 'w') as tmp_file:
            tmp_file.write(csv)

        xcols = ['x1', 'x3', 'x2']
        ycols = ['y2']
        wcol = 'w'
        self.ds = mds.Dataset(X=self.df1[xcols].values,
                              Y=self.df1[ycols].values,
                              W=self.df1[wcol].values,
                              x_labels=xcols,
                              y_labels=ycols)
        self.all_cols = xcols + ycols + [wcol]  


    def tearDown(self):
        """
        Do some cleanup
        """
        os.remove(self.TMP_FILENAME)

    def test_csv(self):
        ds = mds.Dataset.from_csv(self.TMP_FILENAME,
                                  x_labels=['b','a'],
                                  y_labels=['c'])

        self.assertTrue(ds.X.shape == (3, 2))
        self.assertTrue(ds.Y.shape == (3, 1))
        self.assertTrue(ds.W is None)
        self.assertTrue((ds.XY == [[1, 2, 3],
                                   [5, 6, 7],
                                   [9, 0, 1]]).all())

    def test_constructor(self):
        with self.assertRaises(AssertionError):
            mds.Dataset(X=self.df1[['x1', 'x2']].values,
                        Y=self.df1[['y1']].values[:-1])
        with self.assertRaises(AssertionError):
            mds.Dataset(X=self.df1[['x1', 'x2']].values,
                        Y=self.df1[['y1']].values,
                        x_labels=['x1'])
        with self.assertRaises(AssertionError):
            mds.Dataset(X=self.df1[['x1', 'x2']].values,
                        Y=self.df1[['y1']].values,
                        W=self.df1['w'].values[:-1])

        
        data = self.ds.data

        for i, (col1, col2) in enumerate(zip(self.all_cols[:-1], self.all_cols[1:])):
            i1 = list(self.df1.columns).index(col1)
            i2 = list(self.df1.columns).index(col2)
            self.assertTrue(np.allclose(data[:,i] * 10**i1, data[:,i+1] * 10**i2))

    def test_tf(self):
        for make_tf_ds in [True, False]:
            tf_full, = self.ds.get_tf(len(self.ds.data), make_tf_ds=make_tf_ds)
            tf_x, tf_y, tf_yw, tf_full_, tf_xy, tf_w = \
                    self.ds.get_tf(
                                len(self.ds.data),
                                ['X', 'Y', 'YW', 'all', 'XY', "W"],
                                make_tf_ds=make_tf_ds
                            )

            with tf.Session() as sess:
                f, f_, x, y, xy, yw, w = \
                    sess.run([tf_full, tf_full_, tf_x, tf_y, tf_xy, tf_yw, tf_w])

            nx = len(self.ds.x_labels)
            i0 = np.argsort(f [:,0])
            i1 = np.argsort(f_[:,0])
            self.assertTrue(np.allclose(f [i0], self.ds.data       ))
            self.assertTrue(np.allclose(f_[i1], self.ds.data       ))
            self.assertTrue(np.allclose(x [i1], self.ds.X          ))
            self.assertTrue(np.allclose(y [i1], self.ds.Y          ))
            self.assertTrue(np.allclose(xy[i1], self.ds.XY         ))
            self.assertTrue(np.allclose(yw[i1], self.ds.data[:,nx:]))
            self.assertTrue(np.allclose(w [i1], self.ds.W          ))