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
| normalization   | 90x320x1 grayscale image                                   | 22 |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 86x316x24               | 23 |
| RELU            |                                                            | 23 |
| MaxPooling      | 2x2 stride, 2x2 pool size, valid padding, outputs 43,158,24| 24 |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 39x154x36               | 25 |
| RELU            |                                                            | 25 |
| MaxPooling      | 2x2 stride, 2x2 pool size, valid padding, outputs 19x77x36 | 26 |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 15x73x48                | 27 |
| RELU            |                                                            | 27 |
| MaxPooling      | 2x2 stride, 2x2 pool size, valid padding, outputs 7x36x48  | 28 |
| Dropout         | 0.9 keep probability                                       | 30 |
| Convolution 3x3 | 1x1 stride, valid padding, outputs 5x4x64                  | 31 |
| RELU            |                                                            | 31 |
| MaxPooling      | 2x2 stride, 2x2 pool size, valid padding, outputs 2x17x64  | 32 |
| Convolution 2x2 | 1x1 stride, valid padding, outputs 1x16x64                 | 33 |
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

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 102). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 95).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving clockwise and counterclockwise, and driving on both traks.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from a similar version of the [NVIDIA architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

My first step was to pre-process data to transfor it to grayscale and normilize it. This is a good strategy as grayscale offers a simpler task to learn edges and the structure of the road.

Then I took NVIDIA architecture and updated it a little bit, first I reduce conv layers stride and add max pooling layers. To combat the overfitting, I add a few dropout layers. Finally I remove one of the fully connected layers as the output was bigger than the input size, and it could be unnecesary.

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

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8 and batch size 64. I used an adam optimizer so that manually training the learning rate wasn't necessary.
