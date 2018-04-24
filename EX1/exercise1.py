import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import skimage.feature as feat


def load_image(path):
    file = h5py.File(path, 'r')
    if file != None:
        return file

    print("File can not be loaded")


def visualize_imagesAndLables(imDsets, lblDsets):
    global count
    plt.figure(count, figsize=(20,20))
    count = count + 1

    indices = [1, 3, 5, 2, 4, 6]

    for i in range(3):
        imDsets.append(images1[keys_images1[i]])
        lblDsets.append(labels1[keys_labels1[i]])
        plt.subplot(3,2,indices[i])
        plt.title('image '+str(i+1), fontsize=20)
        plt.imshow(imDsets[i])
        plt.subplot(3,2,indices[i+3])
        plt.title('label '+str(i+1), fontsize=20)
        plt.imshow(lblDsets[i])
 
    plt.suptitle('Images and their related labels', fontsize=30)


def apply_filters(imDsets, lblDsets, sigma, gauss, gaussLaplace, gaussGradientMagnitude, structTensor, hessEigenvalues):
    for i in range(3):
        for j in range(len(sigma)):
            gauss.append(ndi.gaussian_filter(imDsets[i], sigma[j]))
            gaussLaplace.append(ndi.gaussian_laplace(imDsets[i], sigma[j]))
            gaussGradientMagnitude.append(ndi.gaussian_gradient_magnitude(imDsets[i], sigma[j]))

            structTensor.append(feat.structure_tensor(imDsets[i], sigma[j]))
            hessianMat = feat.hessian_matrix(imDsets[i], sigma[j])
            #feat.hessian_matrix_eigvals(hessianMat, Hxy=None, Hyy=None)
            #hessEigenvalues.append(tuple(hxy, hyy, hxx))

    square = np.zeros((5, 5))
    square[2, 2] = 4
    H_elems = feat.hessian_matrix(square, sigma=0.1, order='rc')
    #feat.hessian_matrix_eigvals(H_elems)[0]


def visualize_filters():
    visualize_images(gauss, len(gauss), 'Gaussian Filter')
    visualize_images(gaussLaplace, len(gaussLaplace), 'Gaussian Laplace Filter')
    visualize_images(gaussGradientMagnitude, len(gaussGradientMagnitude), 'Gaussian Gradient Magnitude')


def visualize_images(imDsets, num, title):
    global count
    plt.figure(count, figsize=(20,20))
    count = count + 1

    for i in range(num):
        plt.subplot(3, (num//3)+1, i+1)
        plt.title('image '+str(i+1), fontsize=10)
        plt.imshow(imDsets[i])

    plt.suptitle(title, fontsize=30)
    

if __name__ == "__main__":
    count = 0
    images1 = load_image("images_subject02.h5")
    images2 = load_image("images_subject10.h5")
    labels1 = load_image("labels_subject02.h5")
    labels2 = load_image("labels_subject10.h5")
    
    keys_images1 = list(images1.keys())
    keys_images2 = list(images2.keys())
    keys_labels1 = list(labels1.keys())
    keys_labels2 = list(labels2.keys())
   

    imDsets = []
    lblDsets = []

    sigma = [0.7, 1, 1.6, 3.5, 5, 10]

    gauss = []
    gaussLaplace = []
    gaussGradientMagnitude = []

    structTensor = []
    hessEigenvalues = []

    visualize_imagesAndLables(imDsets, lblDsets)
    apply_filters(imDsets, lblDsets, sigma, gauss, gaussLaplace, gaussGradientMagnitude, structTensor, hessEigenvalues)
    visualize_filters()

    plt.show()