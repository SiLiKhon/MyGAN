import os

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

import os.path
import time

import numpy as np
import tensorflow as tf

from MyGAN import dataset as mds
from MyGAN.cramer_gan import CramerGAN
from MyGAN import tf_monitoring as tfmon
from MyGAN.nns import deep_wide_generator, deep_wide_discriminator, noise_layer
from MyGAN.train_utils import adversarial_train_op_func, create_mode

gpu_options = tf.GPUOptions(allow_growth=True)
tf_config = tf.ConfigProto(gpu_options=gpu_options)

tf.reset_default_graph()


N = 1000000
batch_size = 10000
save_interval_secs = 30 * 60
keep_checkpoint_every_n_hours = 2
validation_interval_steps = 200
summary_path = os.path.join("log"    , "example", "cramer_gan")
weights_dir  = os.path.join("weights", "example", "cramer_gan")


if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir)
weights_file = os.path.join(weights_dir, 'cramer_gan.ckpt')

Y01 = np.random.normal(loc=0.0, scale=1., size=(N, 2)).astype(np.float32)
Y2  = Y01.sum(axis=1).reshape((-1, 1)) / 2 + \
        np.random.normal(loc=0.0, scale=0.01, size=(N, 1)).astype(np.float32)
Y = np.concatenate([Y01, Y2], axis=1)

global_step = tf.train.get_or_create_global_step()
step_op = tf.assign_add(global_step, 1)

mode = create_mode()
gan = CramerGAN(
            generator_func=lambda x, ny: deep_wide_generator(
                                x, ny,
                                depth=7,
                                width=64
                            ),
            discriminator_func=lambda x: deep_wide_discriminator(
                                noise_layer(x, 0.005, mode),
                                depth=7,
                                width=64
                            ),
            train_op_func=lambda gloss, dloss, gvars, dvars: adversarial_train_op_func(
                                gloss, dloss, gvars, dvars,
                                optimizer=tf.train.RMSPropOptimizer(0.00001)
                            ),
            gp_factor=10,
            gp_mode='zero_data_only'
        )

ds = mds.Dataset(
    X=np.ones(shape=(N, 1), dtype=np.float32),
    Y=Y,
    W=((Y[:,0] % 2) > 1).astype(np.float32) * 0.5 + 0.75
)

ds_train, ds_test = ds.split(test_size=0.02)

gan.build_graph(ds_train, ds_test, batch_size, mode)

for i in range(3):
    gan.make_summary_histogram("Y{}".format(i), lambda Y: Y[:,i])

gan.make_summary_histogram(
                'Y2_minus_Y01mean',
                lambda Y: Y[:,2] - tf.reduce_mean(Y[:,:2], axis=1)
            )
gan.make_summary_energy(name='energy_distance_full')
for i in range(3):
    gan.make_summary_energy(name='energy_distance_Y{}'.format(i),
                            projection_func=lambda X, Y: Y[:,i])
gan.make_summary_energy(name='energy_distance_Y2_minus_Y01mean',
                        projection_func=lambda X, Y: Y[:,2] - tf.reduce_mean(Y[:,:2], axis=1))

gan.make_summary_sliced_looped_ks(name='sliced_ks')

train_summary = tf.summary.merge(gan.train_summaries)
val_summary   = tf.summary.merge(gan.test_summaries)

print("Summary path is: {}".format(summary_path))
summary_path_train = os.path.join(summary_path, 'train')
summary_path_test  = os.path.join(summary_path, 'test' )

summary_writer_train = tf.summary.FileWriter(
                                        logdir=summary_path_train,
                                        graph=tf.get_default_graph(),
                                        max_queue=100,
                                        flush_secs=20
                                    )
summary_writer_test = tf.summary.FileWriter(
                                        logdir=summary_path_test,
                                        max_queue=100,
                                        flush_secs=60
                                    )
weights_saver = tf.train.Saver(keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

with tf.Session(config=tf_config) as sess:
    latest_ckpt = tf.train.latest_checkpoint(weights_dir)

    if latest_ckpt is None:
        print("Could not find weights file in {}".format(weights_dir))
        sess.run(tf.global_variables_initializer())
        last_time = 0.0
    else:
        weights_saver.restore(sess, latest_ckpt)
        last_time = time.time()

    try:
        while True:
            _, summary, i = sess.run([gan.train_op, train_summary, global_step])
            summary_writer_train.add_summary(summary, i)
            if i % validation_interval_steps == 0:
                summary = sess.run(val_summary, {gan.mode : 'test'})
                summary_writer_test.add_summary(summary, i)
                print("step {}".format(i))
            cur_time = time.time()
            if cur_time - last_time >= save_interval_secs:
                last_time = cur_time
                weights_saver.save(sess, weights_file, global_step=i, write_meta_graph=False)
            sess.run(step_op)
    except KeyboardInterrupt:
        pass

    print("Exiting")
