from io import StringIO

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import tensorflow as tf

from MyGAN.dataset import Dataset

print("Basic tests")

TEST_SIZE: int = 2
df = pd.DataFrame({i : np.arange(1, TEST_SIZE + 1)
                        for i in ['x1', 'x2', 'x3', 'y1', 'y2', 'w']})

df['x2'] *= 10
df['x3'] *= 100
df['y1'] *= 1000
df['y2'] *= 10000
df['w' ] *= 100000

ds = Dataset(X=df[['x1', 'x2', 'x3']].values,
             Y=df[['y1', 'y2']].values,
             W=df['w'].values,
             x_labels=["1", "2", "-2"])

ds2 = Dataset(X=df[['y1', 'x2', 'y2']].values,
             Y=df[['y1', 'y2']].values,
             W=df['w'].values,
             y_labels=["1", "2"])

print(ds)
print(ds.check_similar(ds))

ds = Dataset.concatenate([ds, ds2])

print(ds)

ds3, ds4 = ds.split()
print("\nds3:")
print(ds3)
print("\nds4")
print(ds4)

csv = r"""a,b,c,d
1,2,3,4
5,6,7,8
9,0,1,2
"""

print(Dataset.from_csv(StringIO(csv), x_labels=['a','b'], y_labels=['c']))

print("trying tf")

tf_full, = ds.get_tf(1)
tf_x, tf_y, tf_yw, tf_full_, tf_xy, tf_w = ds.get_tf(1, ['X', 'Y', 'YW', 'all', 'XY', "W"])

with tf.Session() as sess:
    f, f_, x, y, xy, yw, w = sess.run([tf_full, tf_full_, tf_x, tf_y, tf_xy, tf_yw, tf_w])
    for obj, name in zip([f, f_, x, y, xy, yw, w], ['f', 'f_', 'x', 'y', 'xy', 'yw', 'w']):
        print(name)
        print(obj)
        print('')
