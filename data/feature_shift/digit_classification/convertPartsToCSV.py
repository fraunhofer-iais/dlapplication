'''
Created on 18.11.2020

@author: Michael
'''

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os

paths = ["MNIST", "MNIST_M", "SVHN", "SynthDigits", "USPS"]

channels = {"MNIST" : 1, "MNIST_M" : 3, "SVHN" : 3, "SynthDigits" : 3, "USPS" : 1}
modes = {"MNIST" : 'L', "MNIST_M" : 'RGB', "SVHN" : 'RGB', "SynthDigits" : 'RGB', "USPS" : 'L'}

transforms = {
    "MNIST" : transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),

    "SVHN" : transforms.Compose([
        transforms.Resize([28,28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "USPS" : transforms.Compose([
        transforms.Resize([28,28]),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),

    "SynthDigits" : transforms.Compose([
        transforms.Resize([28,28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    
    "MNIST_M" : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    }
for path in paths:
    images, labels = None, None
    print(path,"...")
    for i in range(0,10):        
        imgs, ls = np.load(os.path.join(path, 'partitions/train_part{}.pkl'.format(i)), allow_pickle=True)
        transformed_imgs = []
        for img in imgs:
            image = Image.fromarray(img, mode=modes[path])
            image = transforms[path](image)
            transformed_imgs.append(np.array(image))
        transformed_imgs = np.array(transformed_imgs)
        if i == 0:
            images = transformed_imgs
            labels = ls
        else:
            images = np.concatenate([images,transformed_imgs], axis=0)
            labels = np.concatenate([labels,ls], axis=0)
    data = []
    for i in range(len(labels)):
        val = np.append(np.array(labels[i]), images[i].flatten())
        data.append(val)
    data = np.array(data)
    print(images.shape, labels.shape, data.shape)
    np.savetxt(path+"/"+path+".csv", data, delimiter=",")
        
        
        