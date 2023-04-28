import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
from tensorflow.image import resize_with_crop_or_pad

import matplotlib.pyplot as plt


def plot_history(history):
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()

def plot_result(image_clear, image_noised, image_reconstructed, max_value=255):
    plt.subplots(1,3, figsize=(15, 15))
    plt.subplot(1,3,1)
    plt.imshow(image_clear)
    plt.title(f'Ground True')
    plt.subplot(1,3,2)
    plt.imshow(image_noised)
    psnr = tf.image.psnr(image_clear, image_noised, max_val=max_value)
    ssim = tf.image.ssim(image_clear.astype('float32'), image_noised.astype('float32'), max_val=max_value)
    plt.title(f'Noised, PSNR={psnr:.2f}, SSIM={ssim:.2f}')
    plt.subplot(1,3,3)
    plt.imshow(image_reconstructed)
    psnr = tf.image.psnr(image_clear, image_reconstructed, max_val=max_value)
    ssim = tf.image.ssim(image_clear.astype('float32'), image_reconstructed.astype('float32'), max_val=max_value)
    plt.title(f'Reconstructed, PSNR={psnr:.2f}, SSIM={ssim:.2f}')
    
def calc_mean_image_metrics(images_clear, images_reconstructed, max_value):
    psnrs = []
    ssims = []
    for i in range(images_reconstructed.shape[0]):
        psnr = tf.image.psnr(images_clear[i], images_reconstructed[i], max_val=max_value)
        psnrs.append(psnr)
        ssim = tf.image.ssim(images_clear[i], images_reconstructed[i], max_val=max_value)
        ssims.append(ssim)
    return np.mean(np.array(psnrs)), np.mean(np.array(ssims))

def predict_all(model, x):
    yy = []
    for i in range(x.shape[0]):
        image = x[i][np.newaxis, :]
        y = model.predict(image)
        yy.append(y[0])
    return np.array(yy)

class cut_build():
    
    def __init__(self, source_image, work_shape=(256,256,3)):
        self.source_image = source_image
        self.work_shape = work_shape
        self._crop_resize()
        #self.cropped_image = None
        self.destlist = []
        self.input_shape = self.cropped_image.shape
        
    def _crop_resize(self):
        self.size_new = tuple((np.array(self.source_image.shape) // np.array(self.work_shape)) * np.array(self.work_shape))
        self.cropped_image = np.array(resize_with_crop_or_pad(self.source_image,
                                                     target_height=self.size_new[0],
                                                     target_width=self.size_new[1]
                                                    )
                                     )

    def crop_resize(self, image_in):
        return np.array(resize_with_crop_or_pad(image_in,
                                                     target_height=self.size_new[0],
                                                     target_width=self.size_new[1]
                                                    )
                                     )
	
        
    def cut(self):
        self.qtn = (np.array(self.input_shape) / np.array(self.work_shape)).astype('int')
        self.step_w = self.work_shape[0]
        self.step_h = self.work_shape[1]
        i=0
        j=0
        for i in range(self.qtn[0]):
            for j in range(self.qtn[1]):
                sub = self.cropped_image[i*self.step_w:i*self.step_w+self.step_w,
                                         j*self.step_h:j*self.step_h+self.step_h]
                self.destlist.append(sub)
        return np.array(self.destlist)
                
    def buid(self, pathes):
        out = np.zeros(self.input_shape, dtype='float32')
        i = 0
        j = 0
        k=0
        for i in range(self.qtn[0]):
            for j in range(self.qtn[1]):
                out[i*self.step_w:i*self.step_w+self.step_w,
                    j*self.step_h:j*self.step_h+self.step_h] = pathes[k] #self.destlist[k]
                k+=1
        return out
