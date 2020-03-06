import sys
sys.path.append("../../../../../dlapplication-dev")
sys.path.append("../../../../../dlplatform-dev")

from environments.local_environment import Experiment
from environments.datasources import FileDataSourceFactory
from environments.datasources.dataDecoders.kerasDataDecoders import MNISTDecoder
from utils.models.keras.MNISTNetwork import MNISTCNNNetwork
from DLplatform.synchronizing import DynamicHedgeSync
from DLplatform.aggregating import Average
from DLplatform.learning.factories.kerasLearnerFactory import KerasLearnerFactory
from DLplatform.stopping import MaxAmountExamples
from DLplatform.coordinator import InitializationHandler

if __name__ == "__main__":
    executionMode = 'cpu' # or 'gpu'
    messengerHost = 'localhost'
    messengerPort = 5672
    numberOfNodes = 6
    updateRule = "sgd"
    learningRate = 0.01
    lossFunction = "categorical_crossentropy"
    batchSize = 1
    delta = 0.1
    syncPeriod = 1

    sync = DynamicHedgeSync(delta)
    aggregator = Average()
    stoppingCriterion = MaxAmountExamples(2800)
    dsFactory = FileDataSourceFactory(filename = "../../../../data/textualMNIST/mnist_train.txt", decoder = MNISTDecoder(), numberOfNodes = numberOfNodes, indices = 'roundRobin', shuffle = False, cache = False)
    learnerFactory = KerasLearnerFactory(network=MNISTCNNNetwork(), updateRule=updateRule, learningRate=learningRate, lossFunction=lossFunction, batchSize=batchSize, syncPeriod=syncPeriod, delta=delta)
    initHandler = InitializationHandler()

    exp = Experiment(executionMode = executionMode, messengerHost = messengerHost, messengerPort = messengerPort,
        numberOfNodes = numberOfNodes, sync = sync,
        aggregator = aggregator, learnerFactory = learnerFactory,
        dataSourceFactory = dsFactory, stoppingCriterion = stoppingCriterion, initHandler = initHandler)
    exp.run("MNISTkerasCNN")
