import os
import cv2
import numpy as np

os.chdir('/home/lab/aerialimagecleansing/ImageInpainting/lama/output')
img = cv2.imread('image1_mask001.png', cv2.IMREAD_GRAYSCALE)

img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

gmin = np.min(img)
gmax = np.max(img)
img_norm = np.clip((img - gmin) * 255. / (gmax - gmin), 0, 255).astype(np.uint8)
cv2.imwrite('image1_mask001_norm.png', img_norm)