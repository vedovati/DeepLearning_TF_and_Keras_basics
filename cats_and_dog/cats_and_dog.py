import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X / 255.0

model = Sequential()

# 1th layer
# convolutional layer (64 units to divide the image, window size, input shape)
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
# take the maximum value of the window 
model.add(MaxPooling2D(pool_size=(2,2)))

# 2nd layer
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# 3rd layer
# flatten the model from 2d
model.add(Flatten())
# fully connected layer with 64 neurons
model.add(Dense(64))

# output layer
# 1 because output can be only 1 and 0
model.add(Dense(1))
model.add(Activation('sigmoid'))

# binary cross entropy because only 2 value (1 and 0)
model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1)