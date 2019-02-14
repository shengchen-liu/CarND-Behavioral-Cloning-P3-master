# **Behavioral Cloning** 
### This is Project 4 of Udacity's Self-Driving Car Nanodegree Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---
[//]: # (Image References)

[image0]: ./examples/pipeline.png "Pipeline"
[image1]: ./examples/net.png "Model Visualization"
[image2]: ./examples/data_samples_before_preprocessing.png "before preprocess"
[image3]: ./examples/data_samples_after_preprocessing.png "after preprocess"
[image4]: ./examples/training_data_distribution.png "data distribution"
[image5]: ./examples/nvidia_architecture.PNG "nvidia"
[image_loss]: ./examples/loss.png "train loss"
[image_val]: ./examples/val_loss.png "val Image"
[image_challenge]: ./test_videos_output/project_video.gif "challenge"

To view the video on Youtube, click the following image:

[![IMAGE ALT TEXT HERE](./test_videos_output/project_video.gif)](https://youtu.be/TVTIUdusZGc) 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

![alt text][image0]

[//]: # (Image References)


### The dataset
Data for this task can be gathered with the Udacity simulator itself. Indeed, when the simulator is set to *training mode*, the car is controlled by the human though the keyboard, and frames and steering directions are stored to disk. For those who want to avoid this process, Udacity made also available an "off-the-shelf" training set. For this project, I employed this latter.

Udacity training set is constituted by 8036 samples. For each sample, two main information are provided:
- three frames from the frontal, left and right camera respectively
- the corresponding steering direction

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.hdf5 containing a trained convolution neural network 
* config.py containing hyperparameters of the network
* load_data.py containing the script to load data
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.hdf5 run_1
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Network architecture is borrowed from the aforementioned [NVIDIA paper](https://arxiv.org/pdf/1604.07316v1.pdf) in which they tackle the same problem of steering angle prediction, just in a slightly more unconstrained environment :-)

The architecture is shown below:

![alt_text][image5]

Input normalization is implemented through a `Lambda` layer, which constitutes the first layer of the model. In this way input is standardized such that lie in the range [-1, 1]: of course this works as long as the frame fed to the network is in range [0, 255].

The choice of ELU activation function (instead of more traditional ReLU) come from [this](https://github.com/commaai/research/blob/master/train_steering_model.py) model of CommaAI, which is born for the same task of steering regression. On the contrary, the NVIDIA paper does not explicitly state which activation function they use.

Convolutional layers are followed by 3 fully-connected layers: finally, a last single neuron tries to regress the correct steering value from the features it receives from the previous layers.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 33). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 103-108). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 85).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

One of my favorite components of Tensorflow is Tensorbard.  With Tensorbard, model structure and progress can be easily visualized:

Train loss:

![alt_text][image_loss]

Valid loss:

![alt_text][image_val]