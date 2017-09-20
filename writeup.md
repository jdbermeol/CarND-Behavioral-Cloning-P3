# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* utils.py Includes utility functions to build dataset; and to build dataset generator
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

| Layer                 |     Description    | Code line |
|:---------------------:|:----------------------:|:----------------------:| 
| Input                 | 160x320x3 image  | |
| Crop                 | Crop top and bottom of the image, outputs 90x320x3  | 14 |
| normalization  | 32x32x1 grayscale image | 15 |
| Convolution 5x5         | 2x2 stride, valid padding, outputs 43x158x24  | 16 |
| RELU                    |       | 16 |
| Convolution 5x5         | 2x2 stride, valid padding, outputs 20x77x36  | 17 |
| RELU                    |       | 17 |
| Convolution 5x5         | 2x2 stride, valid padding, outputs 8x37x48  | 18 |
| RELU                    |       | 19 |
| Dropout              | 0.9 keep probability  | 20 |
| Convolution 3x3         | 1x1 stride, valid padding, outputs 6x35x64  | 21 |
| RELU                    |       | 21 |
| Convolution 3x3         | 1x1 stride, valid padding, outputs 4x33x64  | 22 |
| RELU                    |       | 22 |
| Dropout              | 0.9 keep probability  | 24 |
| Flatten                    |     | 25 |
| Fully connected        | outputs 1164   | 26 |
| Fully connected        | outputs 10   | 27 |
| Dropout              | 0.9 keep probability  | 29 |
| Fully connected        | outputs 50    | 30 |
| Fully connected        | outputs 10   | 31 |
| Fully connected        | outputs 1   | 32 |


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 20, 24, 29). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 92). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 85).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving clockwise and counterclockwise, and driving on both traks.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from a small version of the [NVIDIA architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), same number of convolutional layers, but with small number of filters, fewer dense layers with less outputs.

TODO

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 128730 number of data points. I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was TODO. I used an adam optimizer so that manually training the learning rate wasn't necessary.
