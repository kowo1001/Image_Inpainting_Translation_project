import os
import cv2
import numpy as np

DIR = '/home/lab/aerialimagecleansing/ImageInpainting/data/3dlabsData/Target'
SAVE_DIR = '/home/lab/aerialimagecleansing/ImageInpainting/lama/LaMa_test_images_gen'
TRUE_DIR = '/home/lab/aerialimagecleansing/ImageInpainting/data/3dlabsData/1step_data'
TRUE_SAVE_DIR = '/home/lab/aerialimagecleansing/ImageInpainting/lama'
i = 1
for fname in os.listdir(DIR):
    img = cv2.imread(f'{DIR}/{fname}')
    canvas = np.zeros(shape=img.shape, dtype=np.uint8)
    canvas.fill(0)
    canvas[np.where((img == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    cv2.imwrite(f'{SAVE_DIR}/image{i}.png', img)
    cv2.imwrite(f'{SAVE_DIR}/image{i}_mask001.png', canvas)
    for tname in [x for x in os.listdir(TRUE_DIR) if fname[:2] in x]:
        true_img = cv2.imread(f'{TRUE_DIR}/{tname}')
        true_img = cv2.resize(true_img, (img.shape[1], img.shape[0]))
        cv2.imwrite(f'{TRUE_SAVE_DIR}/image{i}.png', true_img)
        
    i = i + 1