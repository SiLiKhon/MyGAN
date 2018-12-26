import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

if __name__ == '__main__':
    import weighted_ks as wks
else:
    from . import np as wks

def run():
    for _ in range(1000):
        x1 = np.random.normal(size=np.random.poisson(lam=100))
        x2 = np.random.normal(size=np.random.poisson(lam=100))
        assert np.allclose(wks.ks_2samp_w(x1, x2), ks_2samp(x1, x2)[0])

    x1 = np.insert(np.random.uniform(size=10), 0, [0.5])
    x2 = np.insert(np.random.uniform(size=5 ), 0, [0.5])

    print("my ks: {}".format(wks.ks_2samp_w(x1, x2)))
    print("scipy: {}".format(ks_2samp(x1, x2)[0]))

    cdf1 = wks.get_ecdf(x1)
    cdf2 = wks.get_ecdf(x2)

    def draw_cdf(x, y, **kwargs):
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

    sample = np.random.normal(size=100000)
    w = np.ones_like(sample)
    w[(sample > 0) & (sample < 1.)] = 0.5

    print("ks: {}".format(wks.ks_2samp_w(sample, sample, None, w)))

    fig, ax1 = plt.subplots()
    _, bins, _ = ax1.hist(sample, bins=100, alpha=0.5)
    ax1.hist(sample, weights=w, bins=bins, histtype='step')
    cdf1 = wks.get_ecdf(sample)
    cdf2 = wks.get_ecdf(sample, w)

    icdfs = wks._interleave_ecdfs(*cdf1, *cdf2)
    ax2 = ax1.twinx()
    ax2.plot(icdfs[0], icdfs[1], label='normal')
    ax2.plot(icdfs[0], icdfs[2], label='weighted')
    ax2.legend()

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    run()