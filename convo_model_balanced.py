from keras.models import Sequential
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.datasets.mnist import load_data
from keras import backend as K
import numpy as np
import cv2 as cv
from mlxtend.data import loadlocal_mnist

def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image.reshape([28 * 28])

x_train, y_train = loadlocal_mnist(
        images_path='./emnist/emnist-balanced-train-images-idx3-ubyte', 
        labels_path='./emnist/emnist-balanced-train-labels-idx1-ubyte')

x_test, y_test = loadlocal_mnist(
        images_path='./emnist/emnist-balanced-test-images-idx3-ubyte', 
        labels_path='./emnist/emnist-balanced-test-labels-idx1-ubyte')

img_rows, img_cols = 28, 28


x_train = np.apply_along_axis(rotate, 1, x_train)
#y_train = np.apply_along_axis(rotate, 1, y_train)

x_test = np.apply_along_axis(rotate, 1, x_test)
#y_test = np.apply_along_axis(rotate, 1, y_test)

# #load the mnist dataset
# (x_train, y_train), (x_test, y_test)  = load_data(path='mnist.npz')

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# y_train = keras.utils.to_categorical(y_train,10)
# y_test = keras.utils.to_categorical(y_test,10)

#create the sequential model
model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(12, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(800, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(62, activation='softmax'))

# optimizer = optimizers.Adam(
#     lr=0.001,
#     beta_1=0.9, 
#     beta_2=0.999, 
#     epsilon=None, 
#     decay=0.0, 
#     amsgrad=False
# )

for layer in model.layers:
    print(layer.output_shape)

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)
model.fit(x_train, y_train, batch_size=64, epochs=5, shuffle=False, verbose=1)

accuracy = model.evaluate(x=x_test,y=y_test,batch_size=64)
print("Accuracy: ",accuracy[1])

model.save('model_balanced2.h5', overwrite=True)

# img = cv.imread('9.png')
# img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# img_gray = cv.resize(img_gray, (28, 28))
# img_gray = cv.fastNlMeansDenoising(img_gray)
# ret,binImg = cv.threshold(img_gray,127,255,cv.THRESH_BINARY_INV)

# #Make prediction
# prediction = model.predict_classes(binImg.reshape(1, 784))
# print(prediction)