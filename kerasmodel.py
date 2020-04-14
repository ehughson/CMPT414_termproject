import sys
import numpy as np
from matplotlib import pyplot
import os
import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
#from keras.utils import to_categorical
from tqdm import tqdm
from random import shuffle
from scipy.special import expit

TRAIN_DIR = '../kcnnmodel/train'
TEST_DIR = '../kcnnmodel/test'
IMG_SIZE = 50

def label_generator(img):
	word_label = img.split('.')[0]

	if word_label == 'cat': 
		#print("cat")
		return 1
	elif word_label == 'dog': 
		#print("dog")
		return 0

def create_training_data():
	training_data =[]
	tally = 0
	for img in tqdm(os.listdir(TRAIN_DIR)):
		tally = tally + 1
		label = label_generator(img)
		path = os.path.join(TRAIN_DIR,img)
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		resized_image = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
		normalized_image = (resized_image/255)-0.5
		training_data.append([normalized_image, label])
	np.save('train_data.npy', training_data)
	return training_data

def create_testing_data():
	testing_data = []
	tally = 0
	for img in tqdm(os.listdir(TEST_DIR)):
		tally = tally + 1
		path = os.path.join(TEST_DIR,img)
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		resized_image = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
		original_image = resized_image
		normalized_image = (resized_image/255)-0.5
		testing_data.append([normalized_image, original_image])
	#shuffle(testing_data)
	np.save('test_data.npy', testing_data)
	return testing_data


#training CNN 
print ('\n--- training CNN ---')
print('\n*preparing labeled data*')
train_data = create_training_data()
print('*done preparing labeled data*\n')

shuffle(train_data)

train = train_data[:-17000] 
train_X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #features or images
train_y = [i[1] for i in train] #labels
#use fit() to train model. use fit_generator if imageDataGenerator is used

shuffle(train_data)

test = train_data[:-20000]
test_X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #features or images
test_y = [i[1] for i in train] #labels

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(50,50,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))	#sigmoid for binary output, softmax for multiclass

print ('\n--- testing CNN ---')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=3)


#test CNN 
print ('\n--- visualize CNN output ---')
print('\n*preparing unlabeled data*')
view_test = create_testing_data()
print('*preparing unlabeled data*\n')

shuffle(view_test)
view_images = view_test[:9]
image_X = np.array([i[0] for i in view_images]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #features or images
original_image = [i[1] for i in view_images]

predictions = model.predict(image_X)

for j, (im, orig) in enumerate(zip(image_X, original_image)):
	pyplot.subplot(330+1+j)
	pyplot.imshow(orig, cmap='gray')
	pyplot.axis('off')
	cat = "%.2f" % (predictions[j]*100)
	dog = "%.2f" % (100 - (predictions[j]*100))
	string = cat + '%cat, '+ dog + '%dog'
	pyplot.title(string, fontsize = 8)

pyplot.show()

