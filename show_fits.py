from astropy.io import fits
import matplotlib.pyplot as plt
import aplpy
import numpy as np
from skimage.transform import resize

# filename = "J000530.07+002241.1.fits"
image_file = fits.open("FITS_300/BENT/J121447.52+105556.9_clipped.fits")
image = image_file[0].data
# X = np.load("X_test2.npy")
# # y = np.load("y_train.npy")
# i = [0,100,200,319, 50, 250, 290,10]
# for k, d in zip(y,range(y.shape[0])):
# 	if k.argmax() == 3:
# 		print(d)
print(image.shape)
# for ii in i:
# 	image_data = X[ii,:,:]
# 	print(image_data.shape)
# 	print(y[ii,:])
# 	# a = resize(image_data,(300,300))
plt.figure()
plt.imshow(image,cmap="jet")
plt.colorbar()
plt.show()

# plt.figure()
# plt.imshow(a,cmap="jet")
# plt.colorbar()
# plt.show()

# mean = np.average(a)
# threshold = np.std(a)*0
# above = np.where(a > mean+threshold,a,0)

# plt.figure()
# plt.imshow(above,cmap="jet")
# plt.colorbar()
# plt.show()
