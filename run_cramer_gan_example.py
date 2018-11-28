import os

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'

import os.path

import numpy as np
import tensorflow as tf

from MyGAN import dataset as mds
from MyGAN.cramer_gan import CramerGAN
from MyGAN import tf_monitoring as tfmon
from MyGAN.nns import deep_wide_generator, deep_wide_discriminator
from MyGAN.train_utils import adversarial_train_op_func

tf.reset_default_graph()


N = 1000000
batch_size = 100000
summary_path = os.path.join("log", "example", "cramer_gan")

Y01 = np.random.normal(loc=0.0, scale=1., size=(N, 2)).astype(np.float32)
Y2  = Y01.sum(axis=1).reshape((-1, 1)) / 2 + \
        np.random.normal(loc=0.0, scale=0.01, size=(N, 1)).astype(np.float32)
Y = np.concatenate([Y01, Y2], axis=1)

global_step = tf.train.get_or_create_global_step()
step_op = tf.assign_add(global_step, 1)
learning_rate = tf.train.exponential_decay(0.0005, global_step, 100, 0.95)

gan = CramerGAN(
            generator_func=deep_wide_generator,
            discriminator_func=deep_wide_discriminator,
            train_op_func=lambda gloss, dloss, gvars, dvars: adversarial_train_op_func(
                                gloss, dloss, gvars, dvars,
                                optimizer=tf.train.RMSPropOptimizer(learning_rate)
                            ),
            gp_factor=100
        )

ds = mds.Dataset(
    X=np.ones(shape=(N, 1), dtype=np.float32),
    Y=Y
)

ds_train, ds_test = ds.split(test_size=0.2)

gan.build_graph(ds_train, ds_test, batch_size, noise_std=0.1)
hist_summaries = [tfmon.make_histogram(
                            summary_name='Y{}'.format(i),
                            input=gan._generator_output[:,i],
                            reference=gan._Y[:,i],
                            label='Generated',
                            label_ref='Real'
                        )
                  for i in range(3)]

hist_summaries += [
            tfmon.make_histogram(
                    summary_name='Y2_minus_Y01mean',
                    input    =gan._generator_output[:,2] - tf.reduce_mean(gan._generator_output[:,:2], axis=1),
                    reference=gan._Y               [:,2] - tf.reduce_mean(gan._Y               [:,:2], axis=1),
                    label='Generated',
                    label_ref='Real'
                )
        ]

train_summary = tf.summary.merge([
        gan.merged_summary, tf.summary.scalar("Learning_rate", learning_rate)
    ])
val_summary = tf.summary.merge([gan.merged_summary] + hist_summaries)

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

    try:
        while True:
            _, summary, i = sess.run([gan._train_op, train_summary, global_step])
            summary_writer_train.add_summary(summary, i)
            if i % 5 == 0:
                summary = sess.run(val_summary, {gan.mode : 'test'})
                summary_writer_test.add_summary(summary, i)
                print("step {}".format(i))
            
            sess.run(step_op)
    except KeyboardInterrupt:
        pass

    print("Exiting")
