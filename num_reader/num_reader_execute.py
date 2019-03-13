import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sys import argv

scropt, I = argv
i = int(I)

# Take data to predict
# take data from the mnist datasets
mnist = tf.keras.datasets.mnist
# function return the Tuple of Numpy arrays of the datas
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalized data from 0-255 to 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# open a model
new_model = tf.keras.models.load_model('num_reader.model')
# make predictions to a model, predict always want list, return a numpy arry of predictions
predictions = new_model.predict(x_test)

# print a prediction in a better way with numpy
print('the number is: ', np.argmax(predictions[i]))

# print number image
plt.imshow(x_test[i])
plt.show()