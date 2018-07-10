# coding=utf-8

import tensorflow as tf
from keras.backend import tensorflow_backend, set_session


def set_tf():
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)
    set_session(tf.Session(config=config))
