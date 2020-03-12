from DLplatform.learning.factories.kerasLearnerFactory import KerasNetwork
import numpy as np

class ResNet18(KerasNetwork):
    def __init__(self, numClasses, inputShape):
        self.numClasses = numClasses
        self.inputShape = inputShape
        pass
    
    def __call__(self):
        from utils.models.keras.resnet.resnet import ResnetBuilder
        network = ResnetBuilder.build_resnet_18(self.inputShape, self.numClasses)
        return network

    def __str__(self):
        return "ResNet18"
    
class ResNet34(KerasNetwork):
    def __init__(self, numClasses, inputShape):
        self.numClasses = numClasses
        self.inputShape = inputShape
        pass
    
    def __call__(self):
        from utils.models.keras.resnet.resnet import ResnetBuilder
        network = ResnetBuilder.build_resnet_34(self.inputShape, self.numClasses)
        return network

    def __str__(self):
        return "ResNet34"
    
class ResNet50(KerasNetwork):
    def __init__(self, numClasses, inputShape):
        self.numClasses = numClasses
        self.inputShape = inputShape
        pass
    
    def __call__(self):
        from utils.models.keras.resnet.resnet import ResnetBuilder
        network = ResnetBuilder.build_resnet_50(self.inputShape, self.numClasses)
        return network    
    
    def __str__(self):
        return "ResNet50"
    
class Cifar10Resnet18(ResNet18):
    def __init__(self):
        imgRows = 32
        imgCols = 32
        imgChannels = 3
        self.numClasses =  10
        self.inputShape = (imgChannels, imgRows, imgCols)
        
class Cifar10Resnet34(ResNet34):
    def __init__(self):
        imgRows = 32
        imgCols = 32
        imgChannels = 3
        self.numClasses =  10
        self.inputShape = (imgChannels, imgRows, imgCols)
        
class Cifar10Resnet50(ResNet50):
    def __init__(self):
        imgRows = 32
        imgCols = 32
        imgChannels = 3
        self.numClasses =  10
        self.inputShape = (imgChannels, imgRows, imgCols)

class ResNet18_Cifar10_32(ResNet18):
    def __init__(self):
        ResNet18.__init__(self, 10, (3,32,32))

    def __call__(self):
        from utils.models.keras.resnet.resnet import ResnetBuilder
        network = ResnetBuilder.build_resnet_18(self.inputShape, self.numClasses)
        return network

    def __str__(self):
        return "ResNet18 Cifar10 input (3,32,32)"

