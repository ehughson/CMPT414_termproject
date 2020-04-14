Steps to compiling both models:
First download both model folders from dropbox:

kcnnmodel:
https://www.dropbox.com/sh/m41wpevxgyy2gl7/AADK5Yfr8DscnN3LocWy_78Qa?dl=0

scratchCNNmodel:
https://www.dropbox.com/sh/lzd4v8xciq65pyb/AADWy1AUHO7cUD4p0fAX4Crsa?dl=0


1) Download the dataset from https://www.kaggle.com/c/dogs-vs-cats/data
	a) Once this dataset is downloaded make sure that all training images and testing images are together in seperate training and testing directory paths called ../kcnnmodel/train' and '../kcnnmodel/test'' for kcnn training and for the from scratch '../scratchCNNmodel/train' and '../scratchCNNmodel/test'
		Note -- empty folders for train and test folder should be created in each models corresponding folder, you are to put the data from kaggle into their respective folders.


2) Once all the cat and dog training photos are put into their respective folder, you can navigate to the scratchCNNModel executable located in dist folder which contains the executable for the from scratch CNN that was created. 
	Note: this tutorial https://victorzhou.com/blog/intro-to-cnns-part-1/ was used to create the CNN code 

3) The output will take a while before it is fully completed. 

4) after running the scratchCNNModel, you can implement the keras model in the other file folder using the same training and testing data. The keras executable is in the dist folder in the kcnnmodel folder.  

