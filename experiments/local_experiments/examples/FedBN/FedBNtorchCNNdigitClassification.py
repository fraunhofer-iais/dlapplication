import sys
sys.path.append("../../../../../dlapplication-dev")
sys.path.append("../../../../../dlplatform-dev")

from environments.local_environment import Experiment
from environments.datasources import MultiFileDataSourceFactory
from environments.datasources.dataDecoders.pytorchDataDecoders import DigitDatasetsDecoder
from dlutils.models.pytorch.digitClassification import DigitClassificationNN
from DLplatform.synchronizing import PeriodicSync
from DLplatform.aggregating import Average
from DLplatform.learning.factories.pytorchLearnerFactory import PytorchFedBNLearnerFactory
from DLplatform.stopping import MaxAmountExamples
from DLplatform.coordinator import InitializationHandler

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":
    executionMode = 'cpu' # or 'gpu'

    messengerHost = 'localhost'
    messengerPort = 5672
    numberOfNodes = 5
    updateRule = "SGD"
    learningRate = 1e-2
    learningParams = {}
    lossFunction = "CrossEntropyLoss"
    batchSize = 32
    delta = None
    sync = PeriodicSync()
    syncPeriod = 100
    minStartNodes = 5
    minStopNodes = 0
    
    aggregator = Average()
    stoppingCriterion = MaxAmountExamples(2800)
    
    filenames = ["../../../../data/feature_shift/digit_classification/MNIST/MNIST.csv",
                 "../../../../data/feature_shift/digit_classification/MNIST_M/MNIST_M.csv",
                 "../../../../data/feature_shift/digit_classification/SVHN/SVHN.csv",
                 "../../../../data/feature_shift/digit_classification/SynthDigits/SynthDigits.csv",
                 "../../../../data/feature_shift/digit_classification/USPS/USPS.csv",
                 ]
    
    dsFactory = MultiFileDataSourceFactory(filenames = filenames, decoder = DigitDatasetsDecoder(), numberOfNodes = numberOfNodes, indices = 'roundRobin', shuffle = False, cache = False)
    learnerFactory = PytorchFedBNLearnerFactory(network=DigitClassificationNN(), updateRule=updateRule, learningRate=learningRate, learningParams=learningParams, lossFunction=lossFunction, batchSize=batchSize, syncPeriod=syncPeriod)
    initHandler = InitializationHandler()

    exp = Experiment(executionMode = executionMode, messengerHost = messengerHost, messengerPort = messengerPort, 
        numberOfNodes = numberOfNodes, sync = sync, 
        aggregator = aggregator, learnerFactory = learnerFactory, 
        dataSourceFactory = dsFactory, stoppingCriterion = stoppingCriterion, 
        initHandler = initHandler, minStartNodes = minStartNodes, minStopNodes = minStopNodes)
    exp.run("MNISTtorchCNN")

