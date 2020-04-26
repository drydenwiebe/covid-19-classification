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
from torch.utils.data import DataLoader, sampler
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

        # Replace fully connected layer of our model to a 2048 feature vector output
        self.classifier = nn.Sequential(
                      nn.Linear(self.model.fc.in_features, 1024), 
                      nn.ReLU(), 
                      nn.Linear(1024,512),
                      nn.ReLU(),
                      nn.Linear(512, 2),                   
                      nn.LogSoftmax(dim=1))
        
        self.model.fc = self.classifier


    def forward(self, x):
        return self.model(x)

    def fit(self, dataloaders, num_epochs):
            self.optimizer = optim.Adam(self.model.fc.parameters())
            # Reduces our learning by a certain factor when less progress is being made in our training.
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, 4)
            # Loss function
            # criterion = nn.BCELoss()
            criterion = nn.NLLLoss()
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
                        self.optimizer.zero_grad()

                        outputs = self.model.forward(inputs)
                        # outputs = outputs.reshape(labels.shape)
                        #calculates the loss between the output of our model and ground-truth labels
                        loss = criterion(outputs, labels)

                        #backpropagate gradients from the loss node through all the parameters
                        loss.backward()
                        #Update parameters(Weighs and biases) of our model using the gradients.
                        self.optimizer.step()

                        train_loss += loss.item() * inputs.size(0)

                        preds = torch.zeros(outputs.shape[0], dtype=float)
                        _, preds = torch.max(outputs, 1)

                        train_acc += torch.sum(preds == labels.data)

                        size = len(dataloaders['train'])

                        print(f'Epoch: {epoch}\t{100 * (j + 1) / size:.2f}% complete.\n', end='\r')

                # Do scheduler step after learning rate step
                scheduler.step()
                
                # Set to evaluation mode
                self.model.eval()

                # Validation phase
                with torch.no_grad():

                    for inputs, labels in dataloaders['valid']:
                        outputs = self.model.forward(inputs)
                        # outputs = outputs.reshape(labels.shape)
                        #calculates the loss between the output of our model and ground-truth labels
                        loss = criterion(outputs, labels)

                        valid_loss += loss.item() * inputs.size(0)

                        preds = torch.zeros(outputs.shape[0], dtype=float)
                        
                        _, preds = torch.max(outputs, 1)
                        
                        # print('val predictions')
                        # print(preds)
                        # print(labels.data)

                        valid_acc += torch.sum(preds == labels.data)

                    # Calculate average losses
                    train_loss = train_loss / dataset_sizes['train']
                    valid_loss = valid_loss / dataset_sizes['valid']

                    # Calculate average accuracy
                    train_acc = train_acc.item() / dataset_sizes['train']
                    valid_acc = valid_acc.item() / dataset_sizes['valid']

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
        
# def test_on_new(model):
#     model.eval()
#     # print(model)
#     crit = nn.NLLLoss()
#     test_loss = 0.0
#     test_acc = 0
#     for inputs, labels in dataloaders['test']:
#         out = model.forward(inputs)
#         # outputs = outputs.reshape(labels.shape)
#         #calculates the loss between the output of our model and ground-truth labels
#         loss = crit(out, labels)
    
#         test_loss += loss.item() * inputs.size(0)
    
#         pred = torch.zeros(out.shape[0], dtype=float)
        
#         _, pred = torch.max(out, 1)
        
#         print('val predictions')
#         print(pred)
#         print(labels.data)
    
#         test_acc += torch.sum(pred == labels.data)
        
#     # Calculate average losses
#     test_loss = test_loss / dataset_sizes['test']
    
#     # Calculate average accuracy
#     test_acc = test_acc.item() / dataset_sizes['test']
    
#     print(f'Validation Loss: {test_loss:.4f}')
#     print(f'\tValidation Accuracy: {100 * test_acc:.2f}%')

# plt.grid(False)
# plt.imshow(x_train[0,0,:,:], cmap=plt.cm.binary)
# plt.xlabel("Actual: %s" % y_train[0])
# plt.show()


# # We instantiate our model class
# model = Model()

# # Run 10 training epochs on our model
# model_ft = model.fit(dataloaders, 10)
            
path = 'resnet_model1.pth'
            
traindir = 'C:\\Users\\karee\\Downloads\\covid-19-x-ray-10000-images\\dataset\\train'
validdir = 'C:\\Users\\karee\\Downloads\\covid-19-x-ray-10000-images\\dataset\\valid'
testdir = 'C:\\Users\\karee\\Downloads\\covid_images\\test'
            
# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
            
dsets = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'valid':
    datasets.ImageFolder(root=validdir, transform=image_transforms['valid']),
    'test':
    datasets.ImageFolder(root=testdir, transform=image_transforms['test']),
}

# Dataloader iterators, make sure to shuffle
dataloaders = {
    'train': DataLoader(dsets['train'], batch_size=12, shuffle=True),
    'valid': DataLoader(dsets['valid'], batch_size=12, shuffle=True),
    'test': DataLoader(dsets['test'], batch_size=12, shuffle=True)
}

dataset_sizes = {x : len(dsets[x]) for x in ["train","valid","test"]}

# # We instantiate our model class
model_1 = Model()

# # Run 10 training epochs on our model
model_ft = model_1.fit(dataloaders, 2)

# test_on_new(model_ft)

# torch.save(model_ft.state_dict(), path)

# model = model.load_state_dict(torch.load(path))
