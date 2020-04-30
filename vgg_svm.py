from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import pickle
from sklearn.model_selection import train_test_split
from torch.utils import data
import pdb
from sklearn.svm import SVC
from image_aug import augment_images

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Get the vgg net model from torchvision.model library
        self.model = models.vgg16(pretrained=True)

        # Building our classifier
        # Turn off training for layers (since it would take too long to train all of them)
        for params in self.model.parameters():
            params.requires_grad = False

        # Replace fully connected layer of our model to a 2048 feature vector output
        self.model.classifier = nn.Sequential()

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":  
    # Load data
    X = pickle.load(open("train_images_512.pk",'rb'), encoding='bytes').numpy()
    y = pickle.load(open("train_labels_512.pk",'rb'), encoding='bytes').numpy()
    x_test = pickle.load(open("test_images_512.pk",'rb'), encoding='bytes').numpy()

    X = ((X/2)+0.5)*255

    x_test = ((x_test/2)+0.5)*255
        
    # Split data into training and validation
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply image augmentation
    x_train, y_train = augment_images(x_train, y_train, num_augment=10)

    # Transform to torch tensor
    tensor_x_train = torch.tensor(x_train).float()
    tensor_y_train = torch.tensor(y_train).float()
    tensor_x_val = torch.tensor(x_val).float()
    tensor_y_val = torch.tensor(y_val).float()

    # Dataset dictionary
    dsets = {
        "train": data.TensorDataset(tensor_x_train,tensor_y_train),
        "val": data.TensorDataset(tensor_x_val,tensor_y_val)}

    dataloaders = {x : data.DataLoader(dsets[x], batch_size=6, shuffle=True)
                    for x in ['train', 'val']}

    dataset_sizes = {x : len(dsets[x]) for x in ["train","val"]}

    # we instantiate our model class
    model = Model()

    # here is our feature vectors for the training set
    train_features = torch.zeros(0, 25088)

    # here is the label vectors for the training set
    train_labels = torch.zeros(0)
 
    for inputs, labels in dataloaders['train']:
        current_features = model.forward(inputs)
        train_features = torch.cat((train_features, current_features))
        train_labels = torch.cat((train_labels, labels))

     # here is our feature vectors for the validation set
    valid_features = torch.zeros(0, 25088)

    # here is the label vectors for the validaion set
    valid_labels = torch.zeros(0)
 
    for inputs, labels in dataloaders['val']:
        current_features = model.forward(inputs)
        valid_features = torch.cat((valid_features, current_features))
        valid_labels = torch.cat((valid_labels, labels))

    # change training data to numpy arrays
    train_features = train_features.numpy()
    train_labels = train_labels.numpy()

    # change validation data to numpy arrays
    valid_features = valid_features.numpy()
    valid_labels = valid_labels.numpy()

    class_weight = {}
    class_weight[0] = 1
    class_weight[1] = 0.25

    classifier = SVC(C=0.5, gamma='auto', kernel='rbf', class_weight=class_weight)
    classifier.fit(train_features, train_labels)

    # Predict train and validation lables
    print(np.mean(classifier.predict(train_features) != train_labels))
    print(np.mean(classifier.predict(valid_features) != valid_labels))
    print(classifier.predict(valid_features))

    # Predict on test examples
    tensor_x_test = torch.tensor(x_test).float()
    test_features = model.forward(tensor_x_test)
    y_pred = classifier.predict(test_features)
    print(y_pred)