import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage

# read image with cv2
img = cv2.imread('./Images/skin2.jpg')

# =============== Affine Transformation =================

# affine transformation with function
rows, cols, ch = img.shape
# method 1 to create the position matrix
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1, pts2)

# method 2 to create the position matrix 
M1 = cv2.getRotationMatrix2D((cols/2,rows/2), 30, 3)
# print(M)
# transfrom the image with affine matrix
img2 = cv2.warpAffine(img, M, (cols, rows))
img3 = cv2.warpAffine(img, M1, (cols, rows))

# convert image with affine transformation
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ============ add Gaussian noise and salt-and-papper noise ============

# ---- add Gaussian noise ----
img_noise = img1.copy()
img_noise_1 = skimage.util.random_noise(img_noise,mode = 'gaussian',seed=None, clip=True, mean=0, var=0.01)
img_noise_2 = skimage.util.random_noise(img_noise, mode = 'gaussian', seed=None, clip=True, mean=0, var=0.04)
img_noise_3 = skimage.util.random_noise(img_noise, mode = 'gaussian', seed=None, clip=True, mean=0, var = 0.09)

# ---- add salt-and-papper noise ----
img_sp_1 = skimage.util.random_noise()


# print out the images
plt.subplot(141)
plt.imshow(img)
plt.subplot(142)
plt.imshow(img1)
plt.subplot(143)
plt.imshow(img2)
plt.subplot(144)
plt.imshow(img3)
plt.show()

