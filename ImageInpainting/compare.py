import os
import cv2
import numpy as np

RAW_DIR = '/home/lab/aerialimagecleansing/ImageInpainting/lama/LaMa_test_images_gen'
TRUE_DIR = '/home/lab/aerialimagecleansing/ImageInpainting/lama'
COMPARE_DIR = '/home/lab/aerialimagecleansing/ImageInpainting/lama/output_compare'

mask_list = [x for x in os.listdir(RAW_DIR) if 'mask' in x]
for fname in [x for x in os.listdir(RAW_DIR) if 'mask' not in x]:
    raw_img = cv2.imread(f'{RAW_DIR}/{fname}', cv2.IMREAD_GRAYSCALE)
    true_img = cv2.imread(f'{TRUE_DIR}/{fname}', cv2.IMREAD_GRAYSCALE)
    
    rt_compared_img = cv2.absdiff(raw_img, true_img)
    cv2.imwrite(f'{COMPARE_DIR}/rt_compare_{fname}', rt_compared_img)

    for mname in [x for x in mask_list if fname[:-4] in x]:
        pred_img = cv2.imread(f'{TRUE_DIR}/{mname}', cv2.IMREAD_GRAYSCALE)
        pt_compared_img = cv2.absdiff(pred_img, true_img)
        print(f'fname : {TRUE_DIR}/{fname} \n pred :{TRUE_DIR}/{mname}')
        cv2.imwrite(f'{COMPARE_DIR}/pt_compare_{mname}', pt_compared_img)
