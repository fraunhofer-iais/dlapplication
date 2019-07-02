import sys
sys.path.append("../../../../dlplatform-dev")
sys.path.append("../../../../dlapplication-dev")

from environments.distributed_environment import Experiment
from environments.datasources import FileDataSourceFactory
from environments.datasources.dataDecoders.pytorchDataDecoders import CifarDecoder
from utils.models.pytorch.resnet import Cifar10ResNet50
from DLplatform.synchronizing import PeriodicSync
from DLplatform.aggregating import Average
from DLplatform.learning.factories.pytorchLearnerFactory import PytorchLearnerFactory
from DLplatform.stopping import MaxAmountExamples
from DLplatform.coordinator import InitializationHandler

import os

if __name__ == "__main__":
    username = 'ladilova'
    sysPath = "/home/IAIS/ladilova/vwframework"
    nodes = ["dpl02.kdlan.iais.fraunhofer.de"]
    
    
    messengerHost = 'dpl03.kdlan.iais.fraunhofer.de'
    messengerPort = 5672
    messengerUser = 'lina'
    messengerPassword = 'password'
    numberOfNodes = 1
    
    dsFactory = FileDataSourceFactory(filename = "/home/IAIS/ladilova/distributedPL_opensource/dlapplication-dev/data/cifar10/train.csv", decoder = CifarDecoder(), numberOfNodes = numberOfNodes, indices = 'roundRobin', shuffle = False, cache = False)

    sync = PeriodicSync()
    syncPeriod = 1
    delta = None
    aggregator = Average()
    stoppingCriterion = MaxAmountExamples(2800)
    
    updateRule = "SGD"
    learningRate = 0.001
    learningParams = {}
    lossFunction = "CrossEntropyLoss"
    batchSize = 8
    learnerFactory = PytorchLearnerFactory(network=Cifar10ResNet50(), updateRule=updateRule, learningRate=learningRate, learningParams=learningParams, lossFunction=lossFunction, batchSize=batchSize, syncPeriod=syncPeriod, delta=delta)
    initHandler = InitializationHandler()

    expFileName = os.path.join(os.getcwd(), os.path.basename(__file__))
    exp = Experiment(expFileName = expFileName, sysPath = sysPath, messengerHost = messengerHost, messengerPort = messengerPort, messengerUser = messengerUser, 
                     messengerPassword = messengerPassword, sync = sync, 
                     aggregator = aggregator, learnerFactory = learnerFactory, 
                     dataSourceFactory = dsFactory, stoppingCriterion = stoppingCriterion, 
                     initHandler = initHandler)
    
    print()
    exp.run("Distributed-Cifar10_torchCNN", username, nodes)
