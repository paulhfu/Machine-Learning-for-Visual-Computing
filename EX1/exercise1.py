import h5py
import matplotlib.pyplot as plt
import numpy as np

def load_image(path):
    file = h5py.File(path, 'r')
    if file != None:
        return file

    print("File can not be loaded")

if __name__ == "__main__":
    images1 = load_image("images_subject02.h5")
    images2 = load_image("images_subject10.h5")
    labels02 = load_image("labels_subject02.h5")
    labels10 = load_image("labels_subject10.h5")
    
    keys_images1 = list(images1.keys())
    keys_images2 = list(images2.keys())
    
    for i in range(3):
        dset = images1[keys_images1[i]]
        plt.figure()
        plt.imshow(dset)
        plt.show()