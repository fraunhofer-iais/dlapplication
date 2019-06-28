import sys
sys.path.append("../../experiments")
#sys.path.append("../../vwframework")

from distributed_environment import Experiment
from DLplatform.dataprovisioning import MNISTDataSource
from DLplatform.synchronizing import PeriodicSync
from DLplatform.learning import LearnerFactory
from DLplatform.aggregating import Average

import os

if __name__ == "__main__":
    username = 'jsicking'
    nodes = ["dpl02.kdlan.iais.fraunhofer.de","dpl03.kdlan.iais.fraunhofer.de","dpl04.kdlan.iais.fraunhofer.de"]

    messengerHost = 'dpl02.kdlan.iais.fraunhofer.de'
    messengerPort = 5672
    messengerUser = 'lina'
    messengerPassword = 'password'
    numberOfNodes = 1
    updateRule = "sgd"
    learningRate = 0.25
    lossFunction = "categorical_crossentropy"
    batchSize = 10
    sync = PeriodicSync()
    syncPeriod = 1
    delta = None
    #randomLearner = LearnerFactory().getKerasNN(updateRule = updateRule, learningRate = learningRate, lossFunction = lossFunction, batchSize = batchSize, syncPeriod = syncPeriod, delta = delta, dataset='mnist')
    #initialParams = randomLearner.getParameters()
    aggregator = Average()
    learnerCreator = "LearnerFactory().getKerasNN(updateRule = \\'sgd\\', learningRate = 0.25, lossFunction = \\'categorical_crossentropy\\', batchSize = 10, syncPeriod = 1, delta = None, dataset=\\'mnist\\')"
    exp = Experiment(messengerHost = messengerHost, messengerPort = messengerPort, messengerUser = messengerUser, messengerPassword = messengerPassword, 
        sync = sync, aggregator = aggregator, 
        learnerCreator = learnerCreator, dataSourceFunction = 'getMNISTDataSource', indicesFunction = 'roundRobin', shuffleData = True, cacheData = True)
    exp.run("firstExp", username, nodes)
