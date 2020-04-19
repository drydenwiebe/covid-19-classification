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


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Get the ResNet-152 model from torchvision.model library
        self.model = models.resnet152(pretrained=True)
        # Building our classifier
        # Turn off training for layers (since it would take too long to train all of them)
        for params in self.model.parameters():
            params.requires_grad = False
        # Replace fully connected layer of our model with our classifier above
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

        self.fc4 = nn.Linear(16, 2)


    def forward(self, x):
        # x is our input data
        x = self.model(x)
        x = self.Dropout1(self.PRelU1(self.fc1(x)))
        x = self.Dropout2(self.PRelU2(self.fc2(x)))
        x = self.Dropout3(self.PRelU3(self.fc3(x)))
        x = self.fc4(x)
        return x

    def fit(self, dataloaders, num_epochs):
            optimizer = optim.Adam(self.parameters(), lr=1e-5)
            # Reduces our learning by a certain factor when less progress is being made in our training.
            scheduler = optim.lr_scheduler.StepLR(optimizer, 4)
            #criterion is the loss function of our model. we use Negative Log-Likelihood loss because we used  log-softmax as the last layer of our model. We can remove the log-softmax layer and replace the nn.NLLLoss() with nn.CrossEntropyLoss()
            criterion = nn.CrossEntropyLoss()
            since = time.time()
            #model.state_dict() is a dictionary of our model's parameters. What we did here is to deepcopy it and assign it to a variable
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
                        #calculates the loss between the output of our model and ground-truth labels                            
                        loss = criterion(outputs, labels)
                        
                        #backpropagate gradients from the loss node through all the parameters
                        loss.backward()
                        #Update parameters(Weighs and biases) of our model using the gradients.
                        optimizer.step()
                    
                        train_loss += loss.item() * inputs.size(0)
                        
                        _, preds = torch.max(outputs, 1)
                        
                        correct_tensor = preds.eq(labels.data.view_as(preds))
                        # Need to convert correct tensor from int to float to average
                        accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                        # Multiply average accuracy times the number of examples in batch
                        train_acc += accuracy.item() * inputs.size(0)
                        
                        train_acc += torch.sum(preds == labels.data)
                        
                        epoch_loss = train_loss / dataset_sizes['train']
                        epoch_acc = train_acc.double() / dataset_sizes['train']
                        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                              'train', epoch_loss, epoch_acc))
                        
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
                        #calculates the loss between the output of our model and ground-truth labels                            
                        loss = criterion(outputs, labels)
                        valid_loss += loss.item() * inputs.size(0)
                        _, preds = torch.max(outputs, 1)      
                        
                        correct_tensor = preds.eq(labels.data.view_as(preds))
                        accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                        # Multiply average accuracy times the number of examples
                        valid_acc += accuracy.item() * inputs.size(0)
                        
                        
                        # valid_acc += torch.sum(preds == labels.data)
                        
                    # Calculate average losses
                    train_loss = train_loss / dataset_sizes['train']
                    valid_loss = valid_loss / dataset_sizes['val']
    
                    # Calculate average accuracy
                    train_acc = train_acc / dataset_sizes['train']
                    valid_acc = valid_acc / dataset_sizes['val']
                        
                    print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')
                    print(f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%')
                        
                        # epoch_loss = valid_loss / dataset_sizes['val']
                        # epoch_acc = valid_acc.double() / dataset_sizes['val']
                        # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        #     'val', epoch_loss, epoch_acc))
                        
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
        
        
# Load data
X = pickle.load(open("train_images_512.pk",'rb'), encoding='bytes').numpy()
y = pickle.load(open("train_labels_512.pk",'rb'), encoding='bytes').numpy()
x_test = pickle.load(open("test_images_512.pk",'rb'), encoding='bytes').numpy()


# X = np.reshape(X, (70, 512, 512, 3))
X = ((X/2)+0.5)*255

# x_test = np.reshape(x_test, (20, 512, 512, 3))
x_test = ((x_test/2)+0.5)*255

# X_cut = np.zeros((70,3,170,170))

# for i in range(X.shape[0]):
#     X_cut[i] = X[i][:,0:170, 0:170]
    

# x_test_cut = np.zeros((20,3,170,170))

# for i in range(x_test.shape[0]):
#     x_test_cut[i] = x_test[i][:,0:170, 0:170]
    
# Split data into training and validation
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Display a training example and its classification
'''
plt.grid(False)
plt.imshow(x_train[0,0,:,:], cmap=plt.cm.binary)
plt.xlabel("Actual: %s" % y_train[0])
plt.show()
'''

# Transform to torch tensor
tensor_x_train = torch.tensor(x_train).float()
tensor_y_train = torch.tensor(y_train)
tensor_x_val = torch.tensor(x_val).float()
tensor_y_val = torch.tensor(y_val)

# Dataset dictionary
dsets = {
    "train": data.TensorDataset(tensor_x_train,tensor_y_train),
    "val": data.TensorDataset(tensor_x_val,tensor_y_val)}

dataloaders = {x : data.DataLoader(dsets[x], batch_size=12, shuffle=True)
                for x in ['train', 'val']}

dataset_sizes = {x : len(dsets[x]) for x in ["train","val"]}

#we instantiate our model class
model = Model()
#run 10 training epochs on our model
model_ft = model.fit(dataloaders, 10)

