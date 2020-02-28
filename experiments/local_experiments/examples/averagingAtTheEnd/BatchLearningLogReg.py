import sys
sys.path.append("../../../../../dlapplication-dev")
sys.path.append("../../../../../dlplatform-dev")


from environments.local_environment import Experiment
from environments.datasources.standardDataSourceFactories import SVMLightDataSourceFactory
from DLplatform.aggregating import Average
from DLplatform.synchronizing.aggAtTheEnd import AggregationAtTheEnd
from DLplatform.learning.factories.sklearnBatchLearnerFactory import SklearnBatchLearnerFactory
from DLplatform.learning.batch.sklearnClassifiers import LogisticRegression
from DLplatform.stopping import MaxAmountExamples


if __name__ == "__main__":
  
    messengerHost = 'localhost'
    messengerPort = 5672
    numberOfNodes = 3
    
    regParam = 0.1
    dim = 4 #skin_segmentation has 4 attributes
    learnerFactory = SklearnBatchLearnerFactory(LogisticRegression, {'regParam' : regParam, 'dim' : dim})
    
    dsFactory = SVMLightDataSourceFactory("../../../../data/classification/skin_segmentation.dat", numberOfNodes, indices = 'roundRobin', shuffle = False)
    stoppingCriterion = MaxAmountExamples(100)
        
    aggregator = Average()
    sync = AggregationAtTheEnd()
    
    exp = Experiment(executionMode = 'cpu', messengerHost = messengerHost, messengerPort = messengerPort, 
        numberOfNodes = numberOfNodes, sync = sync, 
        aggregator = aggregator, learnerFactory = learnerFactory, 
        dataSourceFactory = dsFactory, stoppingCriterion = stoppingCriterion, sleepTime=0)
    exp.run("RadonMachine_test")
