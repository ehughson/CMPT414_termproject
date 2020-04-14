import sys
import numpy as np
from matplotlib import pyplot
import os
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from random import shuffle
from scipy.special import expit



num_filters = 2
filters = np.zeros((2,3,3))
filters[0, :, :] = np.array([[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]])
filters[1, :, :] = np.array([[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]])
#filters = np.random.randn(num_filters,3,3)/9

TRAIN_DIR = '../scratchCNNmodel/train'
TEST_DIR = '../scratchCNNmodel/test'
IMG_SIZE = 80

#nodes = 2s
last_input_shape_sf = list()
last_input_sf = list()
#convolution matrix building functions
def conv(input):
	feature_map = np.zeros((input.shape[0] - filters.shape[1] + 1, input.shape[1]- filters.shape[1]+1, num_filters))

	h = input.shape[0]
	w = input.shape[1]

	shape_check = input.shape

	if (len(shape_check)) == 2:
		map_range = 1
	else:
		map_range = input.shape[-1]

	for map_num in range(map_range):
		for i in range(h-2):
			for j in range(w-2):
				if map_range == 1:
					im_region = input[i:(i+3), j:(j+3)] #convolution kernel
				else:
					im_region = input[i:(i+3), j:(j+3), map_num]
				curr_result = im_region * filters
				feature_map[i, j] = np.sum(curr_result, axis = (1,2)) 

	
	cache = (input, filters)
	
	return feature_map, cache

def con_backprop(gradient, cache, lr):
	X = cache[0]
	W = cache[1]
	dW = np.zeros(X.shape)
	prev_h = W.shape[0]
	prev_w = W.shape[1]
	for h in range(prev_h - 2):
		for w in range(prev_w - 2):
			im_region = X[h:(h+3), w:(w+3)]
			#print(im_region.shape)
			for f in range(2):
				newValue = gradient[h,w,f]*im_region
				dW[f] += newValue


	return dW


#creates a 2x2 regions to use for pooling on the image
def pooling_forward(input):
	stride = 2
	size = 2
	cache = input
	output = np.zeros((np.uint16(input.shape[0]//stride), 
					   np.uint16(input.shape[1]//stride),
					   input.shape[2]))

	for map_num in range(input.shape[2]):	#loop through channels
		r2 = 0
		for r in np.arange(0, input.shape[0]-1, stride):
			c2 = 0
			for c in np.arange(0, input.shape[1]-1, stride):
				output[r2, c2, map_num] = np.max([input[r:r+size, c:c+size, map_num]])
				c2 = c2 + 1
			r2 = r2 + 1

	return output, cache


def pooling_backprop(gradient, cache):
	dX = np.zeros(cache.shape)

	stride =2
	for i in range(cache.shape[0]//stride):
		for j in range(cache.shape[1]//stride):
			im_region2 = cache[(i + 2):(i +2 * 2), (j + 2):(j + 2 * 2)]
			h = im_region2.shape[0]
			w = im_region2.shape[1]
			f = im_region2.shape[2]
			amax = np.max(im_region2, axis = (0,1))

			for i2 in range(h):
				for j2 in range(w):
					for f2 in range(f):
						if im_region2[i2, j2, f2] == amax[f2]:
							dX[i*2+i2, j*2+j2, f2] = gradient[i, j, f2]

	return dX


def ReLU(feature_map):
	ouput = np.zeros(feature_map.shape) #initialize ReLU layer
	for map_num in range(feature_map.shape[-1]): #for every feature map
		for r in np.arange(0, feature_map.shape[0]): #0 to size of width
			for c in np.arange(0,feature_map.shape[1]):	#0 to size of length
				ouput[r, c, map_num] = np.max([feature_map[r,c,map_num],0])
	return ouput


def sf_forward(input,weights,biases,check):
	last_input_shape_sf = input.shape

	input = input.flatten()	
	last_input_sf = input
	
	totals = np.dot(input, weights) + biases
	last_totals_sf = totals

	exponential = np.exp(totals, dtype = np.float) 		
	probability = exponential / np.sum(exponential, axis = 0)

	return probability, last_input_sf, last_input_shape_sf, totals,last_totals_sf, weights, biases


	#softmax backward phase. d_L_d_out = dL/dout
def backdrop(d_L_d_out, learn_rate, last_input, last_input_shape, totals, weights, biases):
		for i, gradient in enumerate(d_L_d_out):
			if gradient == 0:
				continue
			
			input_len, nodes = weights.shape

			t_exp = np.exp(totals)
			S = np.sum(t_exp)	#sum all exponential totals

			d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
			d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

			d_t_d_w = last_input 
			d_t_d_b = 1
			d_t_d_inputs = weights

			d_L_d_t = gradient * d_out_d_t

			#gradient of loss with respect to weight, biases, inputs
			d_L_d_w = np.dot(d_t_d_w[np.newaxis].T, d_L_d_t[np.newaxis])
			d_L_d_b = d_L_d_t * d_t_d_b
			d_L_d_inputs = np.dot(d_t_d_inputs, d_L_d_t)

			#update weights and biases
			weights -= learn_rate * d_L_d_w
			biases -= learn_rate * d_L_d_b	#do weights and biases get updated

		
			return d_L_d_inputs.reshape(last_input_shape), weights, biases#reshape because flattened in forward phase


def train_model(im, label, lr, weights, biases):
	#forward
	out, cache = conv((im/255) - 0.5)	
	out = ReLU(out)
	out, cache = conv(out)
	out = ReLU(out)
	out, cache2 = pooling_forward(out)

	out, last_input, last_input_shape, totals, last_totals, forward_weights, forward_biases = sf_forward(out, weights, biases,0)

	loss = -np.log(out[label])
	acc = 1
	if out[label] > 0.5:
		acc = 1
	else:
		acc = 0

	#initial gradient
	gradient = np.zeros(2)	#out from softmax.forward is vector of 2 probabilities
	gradient[label] = -1 / out[label]

	gradient, back_weights, back_biases = backdrop(gradient, lr, last_input, last_input_shape, last_totals, weights, biases)	
	trained_weights = back_weights
	trained_biases = back_biases

	gradient = pooling_backprop(gradient, cache2)
	gradient = con_backprop(gradient,cache, lr)

	
	return loss, acc, gradient, trained_weights, trained_biases

def test_model(image, weights, biases): #what happens to the changed weights and biases?
	out, cache = conv(image)
	out = ReLU(out)
	out, cache2 = pooling_forward(out)
	out, last_input, last_input_shape, totals, tested_weights, tested_biases = sf_forward(out, weights, biases,1) 

	return out #out[0] = cat, out[1] = dog

def label_generator(img):
	word_label = img.split('.')[0]

	if word_label == 'cat': 
		#print("cat")
		return 0
	elif word_label == 'dog': 
		#print("dog")
		return 1

def create_training_data():
	training_data =[]
	tally = 0
	for img in tqdm(os.listdir(TRAIN_DIR)):
		tally = tally + 1
		label = label_generator(img)
		path = os.path.join(TRAIN_DIR,img)
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
		training_data.append([np.array(img), np.array(label)])
	np.save('train_data.npy', training_data)
	return training_data

def create_testing_data():
	testing_data = []
	tally = 0
	for img in tqdm(os.listdir(TEST_DIR)):
		tally = tally + 1
		print(img)
		if(img == '.DS_Store'):
			continue
		path = os.path.join(TEST_DIR,img)
		
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		
		resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
		original_image = resized_img
		testing_data.append([np.array(resized_img), original_image])
	#shuffle(testing_data)
	np.save('test_data.npy', testing_data)
	return testing_data


#training CNN 
print ('\n--- Training CNN ---')

print('\n*creating training data*')
train_data = create_training_data()
print('*done creating training data*\n')

shuffle(train_data)

train = train_data[:-17000] 
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #features or images
y = [i[1] for i in train] #labels

loss = 0
num_correct = 0
weights = np.random.randn(38*38*2,2)/(38*38*2)
biases = np.zeros(2)

#print(X.shape)
for i, (im, label) in enumerate(zip(X, y)):
	image = im[:,:,0]
	l, acc, gradient, trained_weights, trained_biases = train_model(image, label, .005, weights, biases)

	weights = trained_weights
	biases = trained_biases
	loss += l
	num_correct += acc
	if i % 100 == 99:
		print( '[Step %d] Average Loss %.3f | Accuracy: %d%%' % (i+1, loss/100, num_correct))
		loss = 0
		num_correct = 0




#test CNN 
print ('\n--- Testing CNN ---')
print('\n*creating testing data*')
test_data = create_testing_data()
print('*done creating testing data*\n')

shuffle(test_data)

test = test_data[:-12491]
test_image = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #features or images
origimage = [i[1] for i in test]


for j, (im, orig) in enumerate(zip(test_image,origimage)):
	image = im[:,:,0]
	out = test_model(image, weights, biases)
	pyplot.subplot(330+1+j)
	pyplot.imshow(orig, cmap='gray')
	pyplot.axis('off')
	cat = "%.2f" % (out[0]*100)
	dog = "%.2f" % (out[1]*100)
	string = cat + '%cat, '+ dog + '%dog'
	pyplot.title(string, fontsize = 8)

pyplot.show()

print("*safe!*")
