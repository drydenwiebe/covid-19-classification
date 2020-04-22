import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data

from resnet1 import Model
from image_aug import augment_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode', default="train")

    io_args = parser.parse_args()
    mode = io_args.mode

    if mode == "view":
        X = pickle.load(open("train_images_512.pk",'rb'), encoding='bytes').numpy()
        y = pickle.load(open("train_labels_512.pk",'rb'), encoding='bytes').numpy()
        x_test = pickle.load(open("test_images_512.pk",'rb'), encoding='bytes').numpy()

        # Normalizing data
        X = ((X/2)+0.5)*255
        x_test = ((x_test/2)+0.5)*255

        # Split data into training and validation
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        num_sample = 10
        x_s = x_train[:num_sample]
        y_s = y_train[:num_sample]
        x_aug, y_aug = augment_images(x_s, y_s)
        # np.random.shuffle(sample_aug)

        print(x_aug.shape)

        index = 0
        plt.grid(False)

        def toggle_images(event):
            global index

            index += 1

            if index < len(x_aug):
                plt.imshow(x_aug[index])
                plt.xlabel("Actual: %s" % y_aug[index])
                plt.draw()
            else:        
                plt.close()
        
        plt.imshow(x_aug[index])
        plt.xlabel("Actual: %s" % y_aug[index])

        plt.connect('key_press_event', toggle_images)
        plt.show()
    

    elif mode == "train":
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
        # x_train, y_train = augment_images(x_train, y_train, num_augment=3)

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

        dataloaders = {x : data.DataLoader(dsets[x], batch_size=12, shuffle=True)
                        for x in ['train', 'val']}

        dataset_sizes = {x : len(dsets[x]) for x in ["train","val"]}

        # We instantiate our model class
        model = Model()
        # Run 10 training epochs on our model
        model_ft = model.fit(dataloaders, 10, dataset_sizes)



