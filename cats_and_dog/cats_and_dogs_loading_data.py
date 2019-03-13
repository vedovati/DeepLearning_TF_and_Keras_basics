import numpy as np
# to show the images
import matplotlib.pyplot as plt
# iterate throw directory
import os
# do image operations
import cv2

DATADIR = "C:/Users/vedov/Desktop/ASP/python/tf/PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50

'''
# for each animals (dog and cat)
for category in CATEGORIES:
    path = os.path.join(DATADIR, category) # creates path to ctas or dogs
    for img in os.listdir(path): # for each image in the path
        # covert image to array (full path to image, covert image in grayscle because rgb has more size and color is not essential in this case)
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break
    break

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
'''

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # creates path to cats or dogs
        class_num = CATEGORIES.index(category) # take the index of CATEGORIES
        for img in os.listdir(path): # for each image in the path
            try:
                # covert image to array (full path to image, covert image in grayscle because rgb has more size and color is not essential in this case)
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) #resize image to 50x50 px
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

print(len(training_data))

import random
# shuffle the list of datas
random.shuffle(training_data)

X = []
y = []

# scompose training_data to X and y
for features, label in training_data:
    X.append(features)
    y.append(label)

# convert list to a numpy array
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import pickle

# save data in files.pickle
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()