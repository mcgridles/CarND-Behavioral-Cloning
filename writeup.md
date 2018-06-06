# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model starts with a 2D cropping layer and a lambda layer to preprocess and normalize the data respectively (model.py lines 78-79).

The core architecture of the model consists of 9 layers:

1. Convolutional layer with a 1x1x8 filter for feature extraction with RELU activation
2. Convolutional layer with a 5x5x16 filter and RELU activation
3. Max pooling layer with SAME padding
4. Convolutional layer with 5x5x32 filter and RELU activation
5. Max pooling layer with SAME padding
6. Dropout layer
7. Fully connected layer
  - Input: Flattened convolutional layer output
  - Output: 120
8. Fully connected layer
  - Input: 120
  - Output: 84
9. Fully connected layer
  - Input: 84
  - Output: 1

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py line 90).

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 96-100). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 95).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving through specific sections of the track that need extra training.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to modify a model architecture that I have experience with and has been proven to work well.

My first step was to use a convolution neural network model similar to the one I used for the Traffic Sign Classifier project. I thought this model might be appropriate because I know that it performs well at classifying images and I have experience with the pipeline. However, I removed some of the dropout layers to see the baseline performance and so I could experiment with dropout later.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. While the validation loss was always slightly higher than the training loss, it was not so much so that it seemed to be excessively overfitting.

That being said, I experimented with adding a dropout layer, which did seem to help performance in the simulation. I could definitely experiment with adding more dropout layers.

Then I added preprocessing layers to take advantage of some of the things that Keras makes easy.

Other than that I just assembled the data in a similar way to the lessons using generators and OpenCV.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, mostly in the areas where there is just dirt on one side of the track. I believe this is because those areas are a small percentage of the track that they are not trained adequately, so to improve the driving behavior in these cases, I recorded more training data through those areas.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 78-93) consisted of a convolution neural network with the following layers and layer sizes

1. 2D cropping layer to crop each image from 320x160x3 -> 235x160x3
2. Lambda layer to normalize the data with 0 mean
3. Convolutional layer with a 1x1x8 filter for feature extraction with RELU activation
4. Convolutional layer with a 5x5x16 filter and RELU activation
5. Max pooling layer with SAME padding
6. Convolutional layer with 5x5x32 filter and RELU activation
7. Max pooling layer with SAME padding
8. Flatten layer
9. Dropout layer with 0.7 dropout rate
10. Fully connected layer
  - Input: 65120
  - Output: 120
11. Fully connected layer
  - Input: 120
  - Output: 84
12. Fully connected layer
  - Input: 84
  - Output: 1

#### 3. Creation of the Training Set & Training Process

My approach to training data was to do two laps around the track one direction, two laps in the other direction, recovering from the left and right side of the lane in several different spots, and a few extra recordings around the dirt parts of the track. I did the extra dirt recordings because it represents such a small percentage of the track that I did not feel it would be adequately trained to handle those areas without it.

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the center of the lane if it ever starts to drift towards the sides. These images show what a recovery looks:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would make sure the car does not pull more to one side. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had roughly 3700 data points. I then preprocessed this data by cropping the images from 320x160x3 to 235x160x3 and normalizing the data with 0 mean.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the fact that, while the training loss would continue to decrease, the validation loss would not or would barely decrease. This shows that more than 3 epochs causes overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
