import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data

# from resnet1 import Model
from image_aug import augment_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode', default="view")

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

        num_sample = 1
        x_s = x_train[:num_sample]
        y_s = y_train[:num_sample]
        images, labels = augment_images(x_s, y_s, num_augment=30)
        # # np.random.shuffle(sample_aug)

        # select image to view
        i = 0
        images, labels = images[i::num_sample], labels[i::num_sample]     # see augmentations of same img
        images = np.moveaxis(images, 1, -1)                               # set to (N, 512,512,3) displayable shape
        print(images.shape)

        # Cycle through images with any key press
        index = 0
        plt.grid(False)

        def toggle_images(event):
            global index

            index += 1

            if index < len(images):
                plt.imshow(images[index])
                plt.xlabel(f"Image {index+1} - label: {labels[index]}")
                plt.draw()
            else:        
                plt.close()
        
        plt.imshow(images[index])
        plt.xlabel(f"Image {index+1} - label: {labels[index]}")

        plt.connect('key_press_event', toggle_images)
        plt.show()


