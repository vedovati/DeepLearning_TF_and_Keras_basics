import tensorflow as tf

# take data from the mnist datasets
mnist = tf.keras.datasets.mnist

# function return the Tuple of Numpy arrays of the datas x = data, y = real value
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalized data from 0-255 to 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# creation of the model
model = tf.keras.models.Sequential()
# flat layers
model.add(tf.keras.layers.Flatten())
# Dense(neuron in layer, activation function[what is going to make that neuron activates] = Re(ctified) L(inear) (U)nit[GOTO standard function])
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# adding another layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# last layer 10 = number of classification in this case, activation appy the sofmax function
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# parameters for training
# optimaizer = adam (default goto optimaizer), loss = categorical_crossentropy(basic), metrics to tracks
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
# train model epochs = how many times train
model.fit(x_train, y_train, epochs=3)

# test model
val_loss, val_acc = model.evaluate(x_test, y_test)
# print resul of test model
print(val_loss, val_acc)

# save model
model.save('num_reader.model')