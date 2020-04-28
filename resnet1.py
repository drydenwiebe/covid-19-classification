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

from image_aug import augment_images

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Get the ResNet-152 model from torchvision.model library
        self.model = models.resnet152(pretrained=True)

        # Building our classifier
        # Turn off training for layers (since it would take too long to train all of them)
        for params in self.model.parameters():
            params.requires_grad = False

        # Replace fully connected layer of our model to a 2048 feature vector output
        self.model.fc = nn.Sequential()

        # Add custom classifier layers
        self.fc1 = nn.Linear(2048, 256)
        self.Dropout1 = nn.Dropout()
        self.PRelU1 = nn.PReLU()

        self.fc2 = nn.Linear(256, 64)
        self.Dropout2 = nn.Dropout()
        self.PRelU2 = nn.PReLU()

        self.fc3 = nn.Linear(64, 16)
        self.Dropout3 = nn.Dropout()
        self.PRelU3 = nn.PReLU()

        self.fc4 = nn.Linear(16, 1)
        self.Sigmoid = nn.Sigmoid()


    def forward(self, x):
        # x is our input data
        x = self.model(x)
        x = self.Dropout1(self.PRelU1(self.fc1(x)))
        x = self.Dropout2(self.PRelU2(self.fc2(x)))
        x = self.Dropout3(self.PRelU3(self.fc3(x)))
        x = self.Sigmoid(self.fc4(x))
        return x

    def fit(self, dataloaders, num_epochs):
            optimizer = optim.Adam(self.parameters(), lr=1e-5)
            # Reduces our learning by a certain factor when less progress is being made in our training.
            scheduler = optim.lr_scheduler.StepLR(optimizer, 4)
            # Loss function
            criterion = nn.BCELoss()
            since = time.time()
            best_model_wts = copy.deepcopy(self.model.state_dict())
            best_acc = 0.0
            valid_loss_min = np.Inf

            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
                
                train_loss = 0.0
                valid_loss = 0.0
                
                train_acc = 0
                valid_acc = 0
                
                # Set to training
                self.model.train()
                
                for j, (inputs, labels) in enumerate(dataloaders['train']):
                        # clear all gradients since gradients get accumulated after every iteration.
                        optimizer.zero_grad()
                        
                        outputs = self.forward(inputs)
                        outputs = outputs.reshape(labels.shape)
                        print(outputs)
                        print(labels)
                        #calculates the loss between the output of our model and ground-truth labels                            
                        loss = criterion(outputs, labels)
                        
                        #backpropagate gradients from the loss node through all the parameters
                        loss.backward()
                        #Update parameters(Weighs and biases) of our model using the gradients.
                        optimizer.step()
                    
                        train_loss += loss.item() * inputs.size(0)
                        
                        preds = torch.zeros(outputs.shape, dtype=float)

                        for k in range(outputs.shape[0]):
                            if outputs[k] >= 0.5:
                                preds[k] = 1.0

                        train_acc += torch.sum(preds.int() == labels.data.int())

                        size = len(dataloaders['train'])
                        
                        print(f'Epoch: {epoch}\t{100 * (j + 1) / size:.2f}% complete.\n', end='\r')
                
                # Do scheduler step after learning rate step
                scheduler.step()

                # Validation phase
                with torch.no_grad():
                    # Set to evaluation mode
                    self.model.eval()
                    
                    for inputs, labels in dataloaders['val']:                                
                        outputs = self.forward(inputs)
                        outputs = outputs.reshape(labels.shape)
                        print(outputs)
                        print(labels)
                        #calculates the loss between the output of our model and ground-truth labels                            
                        loss = criterion(outputs, labels)

                        valid_loss += loss.item() * inputs.size(0)

                        preds = torch.zeros(outputs.shape, dtype=float)

                        for k in range(outputs.shape[0]):
                            if outputs[k] >= 0.5:
                                preds[k] = 1.0
                        
                        valid_acc += torch.sum(preds.int() == labels.data.int()) 

                    # Calculate average losses
                    train_loss = train_loss / dataset_sizes['train']
                    valid_loss = valid_loss / dataset_sizes['val']

                    # Calculate average accuracy
                    train_acc = train_acc.item() / dataset_sizes['train']
                    valid_acc = valid_acc.item() / dataset_sizes['val']
                        
                    print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')
                    print(f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%')
                        
                    # deep copy the model if we obtain a better validation accuracy than the previous one.
                    if valid_loss < valid_loss_min:
                        valid_loss_min = valid_loss
                        best_acc = valid_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))
            
            # load best model parameters and return it as the final trained model.
            self.model.load_state_dict(best_model_wts)
            return self.model
    

    def predict(self, inputs):
        outputs = self.forward(inputs)
        outputs = outputs.flatten()
        outputs[outputs >= 0.5] = 1.
        outputs[outputs < 0.5] = 0.
        return outputs
        
        
# Load data
X = pickle.load(open("train_images_512.pk",'rb'), encoding='bytes').numpy()
y = pickle.load(open("train_labels_512.pk",'rb'), encoding='bytes').numpy()
x_test = pickle.load(open("test_images_512.pk",'rb'), encoding='bytes').numpy()

# Normalizing data
X = ((X/2)+0.5)*255

x_test = ((x_test/2)+0.5)*255
    
# Split data into training and validation
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply image augmentation
x_train, y_train = augment_images(x_train, y_train, num_augment=4)

# Display a training example and its classification
'''
plt.grid(False)
plt.imshow(x_train[0,0,:,:], cmap=plt.cm.binary)
plt.xlabel("Actual: %s" % y_train[0])
plt.show()
'''

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

# We instantiate our model class
model = Model()

# run 10 training epochs on our model
model_ft = model.fit(dataloaders, 100)

# Predict on test examples
tensor_x_test = torch.tensor(x_test).float()
y_pred = model.predict(tensor_x_test)
print(y_pred)
>>>>>>> d21e46ff667b42e52c98acaabfbb717b6b5767e0
