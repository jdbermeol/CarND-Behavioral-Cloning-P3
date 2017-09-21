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

[image1]: ./writeup_imgs/1.jpg "Clockwise"
[image2]: ./writeup_imgs/2.jpg "Counterclockwise"
[image3]: ./writeup_imgs/3.jpg "Right recovery"
[image4]: ./writeup_imgs/4.jpg "Left recovery"
[image5]: ./writeup_imgs/5.jpg "Flip"

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

#### 1. Model architecture

| Layer           |     Description                                            | Code line |
|:---------------:|:----------------------------------------------------------:|:--:| 
| Input           | 160x320x3 image                                            | |
| Crop            | Crop top and bottom of the image, outputs 90x320x3         | 21 |
| normalization   | 160x320x1 grayscale image                                  | 22 |
| Convolution 5x5 | 1x1 stride, valid padding, outputs                         | 23 |
| RELU            |                                                            | 23 |
| MaxPooling      | 2x2 stride, 2x2 pool size, valid padding, outputs          | 24 |
| Convolution 5x5 | 1x1 stride, valid padding, outputs                         | 25 |
| RELU            |                                                            | 25 |
| MaxPooling      | 2x2 stride, 2x2 pool size, valid padding, outputs          | 26 |
| Convolution 5x5 | 1x1 stride, valid padding, outputs                         | 27 |
| RELU            |                                                            | 27 |
| MaxPooling      | 2x2 stride, 2x2 pool size, valid padding, outputs          | 28 |
| Dropout         | 0.9 keep probability                                       | 30 |
| Convolution 3x3 | 1x1 stride, valid padding, outputs                         | 31 |
| RELU            |                                                            | 31 |
| MaxPooling      | 2x2 stride, 2x2 pool size, valid padding, outputs          | 32 |
| Convolution 2x2 | 1x1 stride, valid padding, outputs                         | 33 |
| RELU            |                                                            | 33 |
| Dropout         | 0.9 keep probability                                       | 35 |
| Flatten         |                                                            | 36 |
| Fully connected | outputs 100                                                | 37 |
| Fully connected | outputs 50                                                 | 38 |
| Dropout         | 0.9 keep probability                                       | 40 |
| Fully connected | outputs 10                                                 | 41 |
| Fully connected | outputs 1                                                  | 42 |


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 30, 35, 40). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line TOD). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 102).

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

![alt text][image1]

Then I drive counterclockwise. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center. These images show what a recovery looks like :

![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images. For example, here is an image that has then been flipped:

![alt text][image5]

After the collection process, I had 64365 number of data points. I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was TODO and batch size 64. I used an adam optimizer so that manually training the learning rate wasn't necessary.
