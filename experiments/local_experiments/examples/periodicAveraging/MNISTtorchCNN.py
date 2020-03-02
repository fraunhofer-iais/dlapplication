import sys
sys.path.append("../../../../../dlapplication-dev")
sys.path.append("../../../../../dlplatform-dev")

from environments.local_environment import Experiment
from environments.datasources import FileDataSourceFactory
from environments.datasources.dataDecoders.pytorchDataDecoders import MNISTDecoder
from utils.models.pytorch.MNISTNetwork import MnistNet
from DLplatform.synchronizing import PeriodicSync
from DLplatform.aggregating import Average
from DLplatform.learning.factories.pytorchLearnerFactory import PytorchLearnerFactory
from DLplatform.stopping import MaxAmountExamples
from DLplatform.coordinator import InitializationHandler

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":
    executionMode = 'cpu' # or 'gpu'

    messengerHost = 'localhost'
    messengerPort = 5672
    numberOfNodes = 5
    updateRule = "SGD"
    learningRate = 0.25
    learningParams = {}
    lossFunction = "CrossEntropyLoss"
    batchSize = 8
    sync = PeriodicSync()
    syncPeriod = 1
    delta = None
    
    aggregator = Average()
    stoppingCriterion = MaxAmountExamples(2800)
    dsFactory = FileDataSourceFactory(filename = "../../../../data/textualMNIST/mnist_train.txt", decoder = MNISTDecoder(), numberOfNodes = numberOfNodes, indices = 'roundRobin', shuffle = False, cache = False)
    learnerFactory = PytorchLearnerFactory(network=MnistNet(), updateRule=updateRule, learningRate=learningRate, learningParams=learningParams, lossFunction=lossFunction, batchSize=batchSize, syncPeriod=syncPeriod, delta=delta)
    initHandler = InitializationHandler()

    exp = Experiment(executionMode = executionMode, messengerHost = messengerHost, messengerPort = messengerPort, 
        numberOfNodes = numberOfNodes, sync = sync, 
        aggregator = aggregator, learnerFactory = learnerFactory, 
        dataSourceFactory = dsFactory, stoppingCriterion = stoppingCriterion, initHandler = initHandler)
    exp.run("MNISTtorchCNN")

