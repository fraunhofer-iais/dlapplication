import numpy as np
from PIL import Image
from environments.datasources.dataDecoders import DataDecoder, CSVDecoder

#         self.filename = '/data/user/jsicking/vwplatform/experiments/data/textualMNIST/mnist_train.csv'
#         #self.filename = '/data/user/ladilova/vwplatform/experiments/data/carla/angles_carla_train.csv'
#         self.filename = '../data/carla/carla_train_town1.csv'
#         self.filename = '../data/loss_surface/small_train_data1_full_norm.txt'
#         self.filename = '../data/cifar10/train.csv'
#         self.filename = '../data/cifar100/train.csv'


class MNISTDecoder(CSVDecoder):    
    def __init__(self, delimiter = ',', labelCol = 0):
        CSVDecoder.__init__(self, delimiter, labelCol)
        
    def __call__(self, line):
        parsed_line = [float(c) for c in line.split(self._delimiter)]
        label = int(parsed_line[self._labelCol])
        parsed_line.pop(self._labelCol)
        image = np.asarray(parsed_line, dtype='float32').reshape(1,28,28) / 255.0
        
        return image, label
    
    def __str__(self):
       return "MNIST from text file for pytorch"
   
class DigitDatasetsDecoder(CSVDecoder):    
    def __init__(self, delimiter = ',', labelCol = 0):
        CSVDecoder.__init__(self, delimiter, labelCol)
        
    def __call__(self, line):
        parsed_line = [float(c) for c in line.split(self._delimiter)]
        label = int(parsed_line[self._labelCol])
        parsed_line.pop(self._labelCol)
        image = np.asarray(parsed_line, dtype='float32').reshape(3,28,28)
        
        return image, label
    
    def __str__(self):
       return "MNIST from text file for pytorch"
    
class CifarDecoder(DataDecoder):
    def __call__(self, line):
        parsed_line = [float(c) for c in line.split("\t")[0].split(',')]
        image = np.asarray(parsed_line, dtype='float32').reshape(3,32,32)
        label = int(line.split("\t")[1].replace("\n", "").replace("r", ""))

        return image, label

    def __str__(self):
        return "CIFAR10 or 100 for pytorch"
    
class CarlaDecoder(DataDecoder):
    def __call__(self, line):
        parsed_line = [float(c) for c in line.split(',')]
        #image = np.asarray(parsed_line[1:], dtype='float32').reshape(1,300,800)
        image = np.transpose(np.asarray(parsed_line[1:], dtype='float32').reshape(170,800,3), axes = [2,0,1]) # we want a 3,170,800 image, i.e., the RGB values on the first axis
        label = [parsed_line[0]]
            
        return image, label

    def __str__(self):
        return "Carla images for DD"

class VectorLabelDecoder(DataDecoder):
    def __init__(self, delimiter, labelDelimiter, labelCol = 1):
        self._delimiter = delimiter
        self._labelDelimiter = labelDelimiter
        self._labelCol = labelCol
        
    def __call__(self, line):
        inp = np.asarray([float(c) for c in line.split(self._labelDelimiter)[1 - self._labelCol].split(self._delimiter)])
        label = np.asarray([float(c) for c in line.split(self._labelDelimiter)[self._labelCol].replace("\n","").split(self._delimiter)])

        return inp, label

    def __str__(self):
        return "Vectors of numbers mapped to vectors of numbers"

'''
if the data file consists of image filenames this decoder allows to read the data
here the label is also image - for example in cases for semantic segmentation
'''
class ImageFilesDecoder(DataDecoder):
    def __call__(self, line):
        filename_image = line.split("\t")[0]
        with open(filename_image, 'rb') as f:
            image = Image.open(f).convert('RGB')
        filename_label = line.split("\t")[1].replace("\n", "").replace("\r", "")
        with open(filename_label, 'rb') as f:
            label = Image.open(f).convert('P')

        return image, label

    def __str__(self):
        return "Images dataset"


