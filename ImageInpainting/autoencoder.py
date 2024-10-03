import numpy as np
import matplotlib.pyplot as plt
import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array

TRAIN_DIR = '/home/lab/aerialimagecleansing/ImageInpainting/data/CropData/raw'
TRAIN_CLEAN_DIR = '/home/lab/aerialimagecleansing/ImageInpainting/data/CropData/clean'

TEST_DIR = '/home/lab/aerialimagecleansing/ImageInpainting/data/CropData/raw'
TEST_CLEAN_DIR = '/home/lab/aerialimagecleansing/ImageInpainting/data/CropData/clean'

def create_dataset(dir):
    emp = np.empty(shape=0)
    for croped_data_fname in os.listdir(dir):
        img_data = load_img(f'{dir}/{croped_data_fname}', target_size = (100, 100), color_mode = 'grayscale')
        img_arr = img_to_array(img_data)
        if emp.size > 0:
            emp = np.append(emp, img_arr.T, axis=0)
        else:
            emp = img_arr.T
    return emp

train_set = create_dataset(TRAIN_DIR)
train_clean_set = create_dataset(TRAIN_CLEAN_DIR)
test_set = create_dataset(TEST_DIR)
test_clean_set = create_dataset(TEST_CLEAN_DIR)

# num_pixels = train_set.shape[1] * train_set.shape[2]
# X_train = train_set.reshape(train_set.shape[0], num_pixels).astype('float32') / 255.
# X_train_clean = train_clean_set.reshape(train_set.shape[0], num_pixels).astype('float32') / 255.
# X_test = test_set.reshape(test_set.shape[0], num_pixels).astype('float32') / 255.
# X_test_clean = test_clean_set.reshape(test_set.shape[0], num_pixels).astype('float32') / 255.

X_train = train_set.astype('float32') / 255.
X_train_clean = train_clean_set.astype('float32') / 255.
X_test = test_set.astype('float32') / 255.
X_test_clean = test_clean_set.astype('float32') / 255.
print(X_train.shape)
print(X_train_clean.shape)
print(X_test.shape)
print(X_test_clean.shape)


# # create model
model = Sequential()

# encoder network
model.add(Conv2D(35, 3, activation= 'relu', padding='same', input_shape = (100,100,1)))
model.add(MaxPooling2D(2, padding= 'same'))
model.add(Conv2D(25, 3, activation= 'relu', padding='same'))
model.add(MaxPooling2D(2, padding= 'same'))
#decoder network
model.add(Conv2D(25, 3, activation= 'relu', padding='same'))
model.add(UpSampling2D(2))
model.add(Conv2D(35, 3, activation= 'relu', padding='same'))
model.add(UpSampling2D(2))
model.add(Conv2D(1,3,activation='sigmoid', padding= 'same'))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

# Training model
model.fit(X_train, X_train_clean, validation_data=(X_test, X_test_clean), epochs=30, batch_size=32)
pred = model.predict(X_train)

print(pred.shape)
X_test_clean = np.reshape(X_test_clean, (300,100,100)) * 255.
pred = np.reshape(pred, (300,100,100)) * 255.
X_test = np.reshape(X_test, (-1,100,100)) * 255.
print(pred)

plt.figure(figsize=(20, 4))
print("Clean Images")
for i in range(10,20,1):
    plt.subplot(2, 10, i+1)
    plt.imshow(X_test_clean[i,:,:], cmap='gray')
plt.savefig(f'/home/lab/3dlabsDenoiser/data/3dlabsData/Clean.png', format = 'png')

plt.figure(figsize=(20, 4))
print("RAW Images")
for i in range(10,20,1):
    plt.subplot(2, 10, i+1)
    plt.imshow(X_test[i,:,:], cmap='gray')
plt.savefig(f'/home/lab/3dlabsDenoiser/data/3dlabsData/RAW.png', format = 'png')

plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i in range(10,20,1):
    plt.subplot(2, 10, i+1)
    plt.imshow(pred[i,:,:], cmap='gray')  
plt.savefig(f'/home/lab/3dlabsDenoiser/data/3dlabsData/pred.png', format = 'png')