import numpy as np

import tensorflow as tf


def psnr_loss(y_true, y_pred):
    return 1/tf.image.psnr(y_true, y_pred, max_val=255)

def ssim_loss(y_true, y_pred):
    return 1/tf.image.ssim(y_true, y_pred, max_val=1)

def ssim_l2(y_true, y_pred):
    ssim = 1. - tf.math.log(tf.image.ssim(y_true, y_pred, max_val=1.))
    l2 = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return ssim + l2