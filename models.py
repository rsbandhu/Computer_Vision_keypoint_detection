## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #input image size (224 X 224 X 1)
        self.conv1 = nn.Conv2d(1,32, 5) 
        #output size (220 X 220 X 32)
        
        # followed by maxpool2d, divide by 2
       
        self.maxpool2d = nn.MaxPool2d((2,2))
        #input size (110 X 110  X 32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        #output size (106 X 106 X 64)
        
        # followed by maxpool2d, divide by 2
        
        #input size (53 X 53  X 64)
        self.conv3 = nn.Conv2d(64, 128, 4)
        #output size (50 X 50 X 128)
        
        self.dropout = nn.Dropout2d(p=0.2)
        
        # followed by maxpool2d, divide by 2
        
        self.fc1 = nn.Linear(128*25*25, 512)
        self.fc2 = nn.Linear(512, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        #Max pooling
        x = self.maxpool2d(F.leaky_relu(self.conv1(x)))
        # if the pooling dimension is square, you can just specify one number
        x = self.dropout(self.maxpool2d(F.leaky_relu(self.conv2(x))))
        x = self.dropout(self.maxpool2d(F.leaky_relu(self.conv3(x))))
        
        #print(x.shape())
        x = x.view(x.size(0), -1)
        
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    def num_features(self, x):
        size = x.size()[1:]
        num_features =1
        
        for s in size:
            num_features *= s
        
        return num_features
