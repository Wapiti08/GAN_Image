import cv2
import numpy as np
import matplotlib.pyplot as plt

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

