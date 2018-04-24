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
    labels1 = load_image("labels_subject02.h5")
    labels2 = load_image("labels_subject10.h5")
    
    keys_images1 = list(images1.keys())
    keys_images2 = list(images2.keys())
    keys_labels1 = list(labels1.keys())
    keys_labels2 = list(labels2.keys())
    
    plt.figure(1, figsize=(20,20))
    indexes = [1,3,5,2,4,6]
    imDsets = []
    lblDsets = []
    for i in range(3):
        imDsets.append(images1[keys_images1[i]])
        lblDsets.append(labels1[keys_labels1[i]])
        plt.subplot(3,2,indexes[i])
        plt.title('image '+str(i+1), fontsize=20)
        plt.imshow(imDsets[i])
        plt.subplot(3,2,indexes[i+3])
        plt.title('label '+str(i+1), fontsize=20)
        plt.imshow(lblDsets[i])


    plt.suptitle('Images and their related labels', fontsize=30)
    plt.show()
