''' 
Training an image classifier on CIFAR10

'''

# imports
import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

# allow the code to run on a GPU if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

'''
1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
'''
########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./cifar-10-batches-py', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# functions to show an image


def imshow(img, label):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.title(label)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))




'''
2. Define a ResNet
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # TODO write the forward pass for the residual block as shown in figure 2
        
        # pass the input x through the first convolutional layer, a first batch-
        # normalization and a ReLU activation        
        out = F.relu(self.bn1(self.conv1(x)))
        # pass the output from this through the second convolutional layer and 
        # a second batch-normalization         
        out = self.bn2(self.conv2(out))        
        # add the input x to this after passing it through the shortcut
        out += self.shortcut(x)
        # pass the result through another ReLU activation
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

learning_rate = 0.00001
for d in range(4):

    net = ResNet152()
    net.to(device)

    '''
    3. Define a loss function
    '''
    #TODO Use a Cross-Entropy Loss and Adam as optimizer
    criterion =  nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), learning_rate)


    '''
    4. Train the network on the training data
    '''
    # Loop over the data iterator, and feed the inputs to the
    # network and optimize.

    n_epochs = 10
    train_loss_history = []
    epoch_loss_history = []

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        epoch_loss = 0
        epoch_loss_history.append(epoch_loss)
        for i, data in enumerate(trainloader, 0):
            # TODO get the inputs
            # import pdb; pdb.set_trace()
            input, target = data[0].to(device), data[1].to(device)
            # TODO zero the parameter gradients
            optimizer.zero_grad()
            # TODO forward + backward + optimize
            out = net.forward(input)
            loss = criterion(out, target)
            epoch_loss += loss.item()
            epoch_loss_history.append(epoch_loss)
            train_loss_history.append(loss.item())
            loss.backward()
            optimizer.step()

    epoch_loss_history.append(0)
    print('Finished Training')


    '''
    5. Test the network on the test data
    '''
    # TODO Evaluate the performance of the network by predicting the class label that the neural network
    # outputs, and checking it against the ground-truth. If the prediction is
    # correct, add the sample to the list of correct predictions.
    # The outputs are energies for the 10 classes.
    # Higher the energy for a class, the more the network
    # thinks that the image is of the particular class.
    # So, you get the predicted class as the index of the highest energy.
    # Compute the overall accuracy and the accuracy for each class.
    passed_img = []
    passed_lbl = []
    failed_img = []
    failed_lbl = []
    test_loss_history = []
    k_p = 0
    k_f = 0
    for i, data in enumerate(testloader, 0):
        input, target = data[0].to(device), data[1].to(device)
        out = net.forward(input)
        loss = criterion(out, target)
        test_loss_history.append(loss.item())
        for k in range(4):
            if data[1][k].item() == torch.argmax(out[k]).item():
                if k_p < 4:
                    passed_img.append(data[0][k].numpy().tolist())
                    k_p += 1
                passed_lbl.append(out[k].cpu().detach().numpy().tolist())
            else:
                if k_f < 4:
                    failed_img.append(data[0][k].numpy().tolist())
                    k_f += 1
                failed_lbl.append(out[k].cpu().detach().numpy().tolist())



    #print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / 10000))

    # Save failed and passed sets as well as the NN
    # torch.save(net.state_dict(), '/home/mlcvss18_7/PaulHiltResNet/NN_lr'+str(learning_rate)+'.dat')

    passed_img = np.array(passed_img)
    passed_lbl = np.array(passed_lbl)
    failed_img = np.array(failed_img)
    failed_lbl = np.array(failed_lbl)
    train_loss_history = np.array(train_loss_history)
    test_loss_history = np.array(test_loss_history)
    epoch_loss_history = np.array(epoch_loss_history)


    train_loss_history.tofile('/home/mlcvss18_7/PaulHiltResNet/train_loss_history_lr'+str(learning_rate)+'.dat')
    epoch_loss_history.tofile('/home/mlcvss18_7/PaulHiltResNet/epoch_loss_history_lr'+str(learning_rate)+'.dat')
    test_loss_history.tofile('/home/mlcvss18_7/PaulHiltResNet/test_loss_history_lr'+str(learning_rate)+'.dat')

    passed_img.tofile('/home/mlcvss18_7/PaulHiltResNet/passed_img_lr'+str(learning_rate)+'.dat')
    passed_lbl.tofile('/home/mlcvss18_7/PaulHiltResNet/passed_lbl_lr'+str(learning_rate)+'.dat')
    failed_img.tofile('/home/mlcvss18_7/PaulHiltResNet/failed_img_lr'+str(learning_rate)+'.dat')
    failed_lbl.tofile('/home/mlcvss18_7/PaulHiltResNet/failed_lbl_lr'+str(learning_rate)+'.dat')
    failed_lbl.tofile('/home/mlcvss18_7/PaulHiltResNet/failed_lbl_lr'+str(learning_rate)+'.dat')

    #the_model = ResNet18()
    #the_model.load_state_dict(torch.load('./NN'))

    #passed_img = np.fromfile('./passed_img.dat', dtype=int)
    learning_rate = learning_rate * 10  # 0.0001 0.001 0.01 0.1
    print('Round '+str(d)+' finished')

print('FINISHED!')

