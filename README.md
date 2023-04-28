# On the use of  deep convolution neural network for image denoising.
## Image denoising based the real noise model 

This work was devoted to the application of deep neural networks to the problems of image noise reduction. In the process of work, several architectures were considered and implemented, the results obtained on each of them were compared; the best result with PSNR 23.51 and SSIM 0.66 on the validation set was shown by a modified version of dnCNN (the original dnCNN with added short connections on internal conv blocks).
An important part of the work was the construction of a noise model based on real data collected during its implementation; the model constructed in this way showed a noise pattern almost identical to the real one, and the neural network trained on this model showed the ability to almost completely restore the original image without any noticeable reduction in detail - the SSIM indicator reached a value of 0.915 on validation, which is a very high value.
Good results were also obtained when applying the network trained on the built noise model to real noisy images taken by various models of digital cameras.

files:
ae.ipynb - Autoencoder
dncnn_res.ipynb - modified dnCNN
dncnn_res_final_real_noise050.ipynb - the final version of "res dnCNN", trained on the real noise model
im_prep.ipynb - nosy images preparing using gauss noise
im_prep_real.ipynb - noisy images preparing using the real noise model
raw_noise_v2.ipyng - the real noise model construction
unet.ipynb - Unet
losses.py - loss functions
utils.py - utils
