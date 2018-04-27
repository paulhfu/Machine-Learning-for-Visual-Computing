import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import skimage.feature as feat
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


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
        plt.subplot(3,2,indices[i])
        plt.title('image '+str(i+1), fontsize=20)
        plt.imshow(imDsets[i])
        plt.subplot(3,2,indices[i+3])
        plt.title('label '+str(i+1), fontsize=20)
        plt.imshow(lblDsets[i])
 
    plt.suptitle('Training images and their related labels', fontsize=30)


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


def visualize_filters():
    visualize_images(gauss, len(gauss), 'Gaussian Filter', sigma)
    visualize_images(gaussLaplace, len(gaussLaplace), 'Gaussian Laplace Filter', sigma)
    visualize_images(gaussGradientMagnitude, len(gaussGradientMagnitude), 'Gaussian Gradient Magnitude', sigma)


def visualize_images(imDsets, num, title, sigma):
    global count
    plt.figure(count, figsize=(20,20))
    count = count + 1
    imgIdx = 0

    for i in range(num):
        if(i%6==0):
            imgIdx += 1
        plt.subplot(4, (num//4)+1, i+1)
        plt.title('Training image: '+str(imgIdx)+'  with sigma='+str(sigma[i%6]), fontsize=10)
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

    for i in range(3):
        imDsets.append(images1[keys_images1[i]])
        lblDsets.append(labels1[keys_labels1[i]])

    sigma = [0.7, 1, 1.6, 3.5, 5, 10]

    gauss = []
    gaussLaplace = []
    gaussGradientMagnitude = []

    structTensor = []
    hessEigenvalues = []

#    visualize_imagesAndLables(imDsets, lblDsets)
    apply_filters(imDsets, lblDsets, sigma, gauss, gaussLaplace, gaussGradientMagnitude, structTensor, hessEigenvalues)
#    visualize_filters()

    ngauss=np.array(gauss)
    ngaussLaplace=np.array(gaussLaplace)
    ngaussGradientMagnitude=np.array(gaussGradientMagnitude)
    nstructTensor=np.array(structTensor)
    nlblDsets=np.array(lblDsets)
    print(nstructTensor.shape)
    input=np.zeros((36,3*768*496))

    for i in range(6):
        input[i][:]=np.append(ngauss[i].reshape((1,len(ngauss[0])*len(ngauss[0][0]))), [ngauss[i+6].reshape((1,len(ngauss[0])*len(ngauss[0][0]))), ngauss[i+12].reshape((1,len(ngauss[0])*len(ngauss[0][0])))])
    for i in range(6):
        input[i+6][:]=np.append(ngaussLaplace[i].reshape((1,len(ngaussLaplace[0])*len(ngaussLaplace[0][0]))), [ngaussLaplace[i+6].reshape((1,len(ngaussLaplace[0])*len(ngaussLaplace[0][0]))), ngaussLaplace[i+12].reshape((1,len(ngaussLaplace[0])*len(ngaussLaplace[0][0])))])
    for i in range(6):
        input[i+12][:]=np.append(ngaussGradientMagnitude[i].reshape((1,len(ngaussGradientMagnitude[0])*len(ngaussGradientMagnitude[0][0]))), [ngaussGradientMagnitude[i+6].reshape((1,len(ngaussGradientMagnitude[0])*len(ngaussGradientMagnitude[0][0]))), ngaussGradientMagnitude[i+12].reshape((1,len(ngaussGradientMagnitude[0])*len(ngaussGradientMagnitude[0][0])))])
    for j in range(3):
        for i in range(6):
            input[i][:]=np.append(nstructTensor[i][j].reshape((1,len(nstructTensor[0][0])*len(nstructTensor[0][0][0]))), [nstructTensor[i+6][j].reshape((1,len(nstructTensor[0][0])*len(nstructTensor[0][0][0]))), nstructTensor[i+12][j].reshape((1,len(nstructTensor[0][0])*len(nstructTensor[0][0][0])))])

    input=np.swapaxes(input, 0, 1)
    output=np.append(nlblDsets[0].reshape((1,len(nlblDsets[0])*len(nlblDsets[0][0]))), [nlblDsets[1].reshape((1,len(nlblDsets[0])*len(nlblDsets[0][0]))), nlblDsets[2].reshape((1,len(nlblDsets[0])*len(nlblDsets[0][0])))])
    
    rf = RandomForestClassifier()
    rf.fit(input,output)
    inputProps=rf.predict_proba(input)

    hf = h5py.File('propsData.h5','w')
    hf.create_dataset('inputProps', data=inputProps)
    hf.close()