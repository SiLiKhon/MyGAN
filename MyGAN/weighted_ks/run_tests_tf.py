import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

if __name__ == '__main__':
    import weighted_ks_tf as wks
else:
    from . import tf as wks

def run():
    with tf.Session() as sess:
        x1 = tf.random_normal(shape=(tf.random_poisson(shape=[], lam=100, dtype='int32'), 1000))
        x2 = tf.random_normal(shape=(tf.random_poisson(shape=[], lam=100, dtype='int32'), 1000))
        np_x1, np_x2, wks_result = sess.run([x1, x2, wks.ks_2samp_w(x1, x2)])
        assert np.allclose(wks_result, [ks_2samp(np_x1[:,i], np_x2[:,i])[0] for i in range(1000)])

        x1 = np.insert(np.random.uniform(size=10), 0, [0.5])
        x2 = np.insert(np.random.uniform(size=5 ), 0, [0.5])

        print("my ks: {}".format(sess.run(wks.ks_2samp_w(x1, x2))[0]))
        print("scipy: {}".format(ks_2samp(x1, x2)[0]))

        x1 = x1[:,np.newaxis]
        x2 = x2[:,np.newaxis]

        cdf1 = wks.get_ecdf(x1)
        cdf2 = wks.get_ecdf(x2)

        def draw_cdf(x, y, **kwargs):
            x, y = sess.run([x, y])
            x = np.squeeze(x)
            y = np.squeeze(y)
            y = np.insert(y, np.arange(1, len(y)), y[:-1])
            y = np.insert(y, 0, [0])
            x = np.insert(x, np.arange(1, len(x)), x[1:])
            x = np.insert(x, 0, x[0])
            plt.plot(x, y, **kwargs)

        icdfs = wks._interleave_ecdfs(*cdf1, *cdf2)

        draw_cdf(*cdf1, label='cdf1', linewidth=1)
        draw_cdf(*cdf2, label='cdf2', linewidth=1)
        draw_cdf(icdfs[0], icdfs[1], label='cdf1 - interleaved', linewidth=4, alpha=0.3)
        draw_cdf(icdfs[0], icdfs[2], label='cdf2 - interleaved', linewidth=4, alpha=0.3)
        plt.legend()
        plt.show()

        sample = np.random.normal(size=[100000, 1])
        w = np.ones_like(sample)
        w[(sample > 0) & (sample < 1.)] = 0.5

        print("ks: {}".format(sess.run(wks.ks_2samp_w(sample, sample, None, w))[0]))

        fig, ax1 = plt.subplots()
        _, bins, _ = ax1.hist(sample, bins=100, alpha=0.5)
        ax1.hist(sample, weights=w, bins=bins, histtype='step')
        cdf1 = wks.get_ecdf(sample)
        cdf2 = wks.get_ecdf(sample, w)

        icdfs = sess.run(wks._interleave_ecdfs(*cdf1, *cdf2))
        ax2 = ax1.twinx()
        ax2.plot(icdfs[0], icdfs[1], label='normal')
        ax2.plot(icdfs[0], icdfs[2], label='weighted')
        ax2.legend()

        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    run()