from matplotlib import pyplot
import cv2
import os
from matplotlib.image import imread
import numpy 
import pandas as pd
import skimage.data
import skimage.io
import skimage.color
import sys
#define location of dataset
​
folder = 'train/'
X = [] #array of pixels to train
Y = [] #target array. cat = 0, dog = 1
​
for i in range(1, 9):
	pyplot.subplot(330+1+i)
	filename = os.path.join(folder, 'dog.'+str(i)+'.jpg')
	#filename = folder + 'dog.' + str(i) + '.jpg'
	image = cv2.imread(filename)
	pyplot.imshow(image)
​
pyplot.show()
​
for i in range(1, 9):
	pyplot.subplot(330+1+i)
	filename = os.path.join(folder, 'cat.'+str(i)+'.jpg')
	#filename = folder + 'dog.' + str(i) + '.jpg'
	image = cv2.imread(filename)
	pyplot.imshow(image)
​
pyplot.show()
​
​
​
​
def conv(img, conv_filter):
	#holding empty convoling results
	#img_rows - filter_rows + 1, img_columns - filter_columns + 1, num filters
	feature_maps = numpy.zeros((img.shape[0] - conv_filter.shape[1] + 1, img.shape[1] - conv_filter.shape[1]+1, conv_filter.shape[0]))
​
​
	#going over the iamge with the convolution
	for filt_num in range(conv_filter.shape[0]):
		print("filter", filt_num+1)
		curr_filter = conv_filter[filt_num, :]
​
		#convolve each image channel with its corresponding chanel in the filters. -- Then sum the results to get the feature map
		#this is if it has more than one channel
		if len(curr_filter.shape) >  2:
			conv_map = conv_(img[:,:, 0], curr_filter[:,:,0])
			for ch_num in range(1, curr_filter.shape[-1]):
				conv_map = conv_map + conv_(img[:,:,ch_num], curr_filter[:, :, ch_num])
		#we just have a single channel which should not happen
		else:
			conv_map = conv_(img, curr_filter)
​
​
		feature_maps[: ,:, filt_num] = conv_map
	
	return feature_maps
​
def conv_(img, conv_filter):
	filter_size = conv_filter.shape[0]
	result = numpy.zeros((img.shape))
​
	for r in numpy.uint16(numpy.arange(filter_size/2.0, img.shape[0] - filter_size/2.0 - 2)):
		for c in numpy.uint16(numpy.arange(filter_size/2.0, img.shape[1] - filter_size/ 2.0 - 2)):
​
			#extract regions of equal size to the filter 
			curr_region = img[r:r+filter_size, c:c+filter_size]
			#element wise multiplication
			curr_result = curr_region * conv_filter
			conv_sum = numpy.sum(curr_result)
			result[r, c] = conv_sum
​
	final_result = result[numpy.uint16(filter_size/2.0): result.shape[0] - numpy.uint16(filter_size/2.0), 
						  numpy.uint16(filter_size/2.0): result.shape[1] - numpy.uint16(filter_size/2.0)]
​
	return final_result
​
###########################################################
filename = os.path.join(folder, 'cat.'+str(1)+'.jpg')
	#filename = folder + 'dog.' + str(i) + '.jpg'
img = skimage.io.imread(filename)
img = skimage.color.rgb2gray(img)
​
filter = numpy.zeros((2,3,3))
filter[0, :, :] = numpy.array([[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]])
filter[1, :, :] = numpy.array([[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]])
print("\n**working with conv layer1**")
feature_map = conv(img,filter)
#pyplot.imshow(feature_map.shape[1], cmap="gray") #imshow expects image data as [height, width, 3]
#pyplot.show()
pyplot.subplot(1,2,1)
pyplot.imshow(feature_map[:,:,0],cmap="gray") #feature map has two channels 
pyplot.subplot(1,2,2)
pyplot.imshow(feature_map[:,:,1],cmap="gray")
pyplot.show()
​
​
#we have 2 channels because we are dealing with black and white images. If we allows RGB then we would have 3 channels
#filter = numpy.zeros((2, 3, 3))
#conv_filters:
	#sobel filters
	# filter[0, :, :] = numpy.array([[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]])
	# filter[1, :, :] = numpy.array([[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]])
​
#layer 1 feature map = numpycnn.conv(img, filter)
