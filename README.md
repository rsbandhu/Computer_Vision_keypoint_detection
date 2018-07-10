# Udacity_Computer_Vision_Proj1_keypoint_detection

## Project Overview

In this project, you’ll combine your knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. Your completed code should be able to look at any image, detect faces, and predict the locations of facial keypoints on each face; examples of these keypoints are displayed below.

![Facial Keypoint Detection][image1]

The project will be broken up into a few main parts in four Python notebooks, **only Notebooks 2 and 3 (and the `models.py` file) will be graded**:

__Notebook 1__ : Loading and Visualizing the Facial Keypoint Data

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__Notebook 3__ : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

__Notebook 4__ : Fun Filters and Keypoint Uses


### Submission Notes
Please see the notes below under each section

### `models.py`

#### Specify the CNN architecture
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  Define a CNN in `models.py`. |  Define a convolutional neural network with at least one convolutional layer, i.e. self.conv1 = nn.Conv2d(1, 32, 5). The network should take in a grayscale, square image. |

### Submission Notes
the NN architecture is defined in models.py file
It has got 3 Convolutional layers followed by 2 fully connected layers

### Notebook 2

'data_transform' has been defined
loss function" SmoothedL1loss
Optimizer: Adam (default settings)

Train:
Training done with several CNN architectures, loss functions, batch size, optimizer

Answer all questions: please see answers in the relevant sections

### Notebook 3

#### Detect faces in a given image
Using Haar cascade to detect faces in a given image
Each face is normalized, resized for key point detection using the trained model above.
Keypoints predicted and displayed on top of each image

