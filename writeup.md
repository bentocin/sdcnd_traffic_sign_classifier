# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/dataset_images.png "Exploratory Images of Dataset"
[image2]: ./report_images/dataset_train_histogram.png "Training Histogram"
[image3]: ./report_images/dataset_validation_histogram.png "Validation Histogram"
[image4]: ./report_images/dataset_testing_histogram.png "Testing Histogram"
[image5]: ./report_images/augmented_images.png "Augmented Images"
[image6]: ./report_images/grayscale_images.png "Grayscale Images"

[image7]: ./test_signs/sign_130.jpg "Speed Limit 130"
[image8]: ./test_signs/sign_one_way.jpg "One Way"
[image9]: ./test_signs/sign_priority.jpg "Priority Road"
[image10]: ./test_signs/sign_right_only.jpg "Turn Right Ahead"
[image11]: ./test_signs/sign_stop.jpg "Stop"

[image12]: ./report_images/layer_visualization.png "Layer Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

And here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy to get a rough overview of the dimensions of the dataset:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

I also used the Pandas library to calculate the occurences of each class.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

![alt text][image1]

The following histograms show the distribution of the classes of the different datasets for training, validaion and training. It can be seen that the classes have equal distributions over the different datasets but within the datasets there are classes that stand out and other that are under represented.

![alt text][image2]

![alt text][image3]

![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided against converting the images to grayscale, although this might eases the learning process for the model as three channels are condensed into just one. A test showed that the model for grayscale images learned faster in comparison to the one with color images. However, this benefit was just marginal.

As a next step, I compared two normalization techniques with each other. The first was "x/255-0.5" which leads to data that is ranging from -0.5 to 0.5 as the pixels have 8 bit values ranging from 0 to 255. The other approah was "x-x.mean / x.std" which is the one that was chosen in the end as it performed better than the other.

I decided to generate additional data because it helps the model to generalize better as it more data during training. A "augmenter" preprocessing function will take the original dataset and add augmented images to it until the wanted sample size is reached.

To add more data to the the data set, I used rotation and translation. These two techniques simulate different positions of the traffic signs within the image and make the model more robust against rotated signs. What could be added in a next step is to also augment different brightness levels.

Here is an example of an some original images and an their augmented counterparts:

![alt text][image5]

The difference between the original data set and the augmented data set is that the augmented dataset contains all of the original data and some augmented data in addition.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		  | 32x32x3 RGB image   							| 
| Convolution 5x5     | 1x1 stride, valid padding, outputs 28x28x16 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| ReLU                |                                       | 
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32   |
| Max pooling         | 2x2 stride, outputs 5x5x32  |
| ReLU                |                                   |
| Fully connected		  | outputs 800     									|
| Dropout             |                                   |
| Dense               | outputs 128                       |
| ReLU                |                                   |
| Dense               | outputs 64                        |
| ReLU                |                                   |
| Dense               | outputs 43                        |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an a learning rate of 0.001 and 50 epochs. A lower learning rate of 0.0001 and 100 epochs has been tested as well but performed worse. The batch size was set to 128. As optimizer the Adam Optimizer was chosen. As the data was partly augmented the amount of augmented data was an additional hyperparameter. It was chosen in a way that the original data combined with the augmented data summed up to a sample size of 60000 images that can be trained on each epoch.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 98.0%
* test set accuracy of 96.4%

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I chose a mixture of well known architecture and iterative approach. I startet with LeNet and took it as a benchmark for my following experiments. Based on LeNet I tested different sizes of layers and added some layers to make the network deeper. I believed LeNet was a good starting point as it worked well for the CIFAR-10 dataset where images also are of the size 32x32. The high training set accuracy and lower validation and test accuracy shows that the model overfits a little bit. But the overall performance is quite good.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10] ![alt text][image11] 

The first image might be difficult to classify because it shows a speed limit (130km/h) sign which is not included in the dataset. This means the model has never seen this sign before during training and thus will try to classify it as a sign it already knows. In fact, the chosen model classifies it as speed limit (30km/h) sign with a probability of 92.4% and speed limit (100km/h) sign with a probability of 5.3%. This is a good result for a class that is not in the dataset.

The other images were all classified correctly with high probabilities for the correct classes. The test images all have good lighting conditions, the traffic signs are centered and shown in good quality. The only difficulties should be the varying background which should be ignored by the model anyway.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	       | Probability              |
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop Sign   									| 99.9%             |
| Priority Road     			| Priority Road 										| 100%    |
| Turn Right Ahead					| Turn Right Ahead								| 70%     |
| Speed Limit (130km/h)	      		| Speed Limit (30km/h)			| 92.4%   |
| No Entry			| No Entry      							|               | 100%    |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 96.4%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the priority road image, the model is very sure that this is a priority road (probability of ~100%). The top five soft max probabilities were

| Probability [%]       |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.0        			| Priority Road   									| 
| 0.0     				| Right-of-way at the next intersection 										|
| 0.0					| Yield											|
| 0.0	      			| Bicycles Crossing					 				|
| 0.0				    | Stop     							|


For the stop sign image, the model is also very sure and classifies it as stop sign with 99.9% probability. The top five soft max probabilites were

| Probability [%]      	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.95        			| Stop   									| 
| 0.04     				| Speed Limit (30km/h) 										|
| 0.01					| Speed Limit (20km/h)											|
| 0.0	      			| Speed Limit (50km/h)					 				|
| 0.0				    | Road Work     							|

For the turn right ahead sign, the top five soft max probabilities were

| Probability [%]     	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70.2        			| Turn Right Ahead  									| 
| 29.2     				| Roundabout Mandatory 										|
| 0.5					| Keep Left											|
| 0.03	      			| Ahead Only					 				|
| 0.02				    | Go Straight Or Left     							|

For the speed limit (130km/h) sign, the top five soft max probabilities were

| Probability [%]      	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 92.4        			| Speed Limit (30km/h)   									| 
| 5.3     				| Speed Limit (100km/h) 							|
| 1.2					| Speed Limit (20km/h)											|
| 1.0	      			| Speed Limit (120km/h)					 				|
| 0.1				    | Speed Limit (50km/h)     							|

For the no entry sign, the top five soft max probabilities were

| Probability [%]      	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.0        			| No Entry   									| 
| 0.0     				| Stop 								|
| 0.0					| No Passing For Vehicles Over 3.5 Metric Tons											|
| 0.0      			| No Passing					 				|
| 0.0				    | Priority Road     							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The following visualization shows the filters of the first convolutional layer.
![alt text][image12]
