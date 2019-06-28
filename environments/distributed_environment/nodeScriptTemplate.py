import sys
import os
sys.path.append('{dlplatform_path}')
sys.path.append('{dlapplication_path}')
from DLplatform.worker import Worker
from DLplatform.communicating import Communicator, RabbitMQComm
from DLplatform.dataprovisioning import IntervalDataScheduler
from DLplatform.learningLogger import LearningLogger
from multiprocessing import Process
import time

{imports}

#just for debugging
# from environments.distributed_environment import Experiment
# from environments.datasources import FileDataSourceFactory
# from environments.datasources.dataDecoders.pytorchDataDecoders import CifarDecoder
# from utils.models.pytorch.MNISTNetwork import MnistNet
# from DLplatform.synchronizing import PeriodicSync
# from DLplatform.aggregating import Average
# from DLplatform.learning.factories.pytorchLearnerFactory import PytorchLearnerFactory
# from DLplatform.stopping import MaxAmountExamples
# from DLplatform.coordinator import InitializationHandler





'''
variables: (to be passed as strings without the '-')
    general:
    dlplatform_path      - path to the DLPlatform folder
    dlapplication_path   - path to the DLApplication folder    
    exp_path             - path to the experiments folder
    imports              - the required imports for the learner and data creation
     
    messaging:
    mHost            -    messaging host
    mPort            -    messaging port
    mUser            -    messaging user name
    mPwd             -    messaging user's password
    uniqueId         -    messaging unique id
     
    learner factory:
    learnerFactoryName      -    name of the specific learner factory class
    learnerFactoryParams    -    string containing the required initialization parameters
 
    data source:
    dataSourceFactoryName      -    name of the specific data source factory class
    dataSourceFactoryParams    -    string containing the required initialization parameters
    dataScheduler              -    data scheduler
     
    stopping:
    stoppingCriterionName      -    name of the specific stopping criterion class
    stoppingCriterionParams    -    string containing the required initialization parameters
     
'''
def createWorker(id, exp_path, dataScheduler, learnerFactory):
    nodeId = str(id)
    w = Worker(nodeId)
    w.setDataScheduler(dataScheduler)
    
    comm = RabbitMQComm(hostname ='{mHost}', port = '{mPort}', user = '{mUser}', password = '{mPwd}', uniqueId = '{uniqueId}')
    os.mkdir(os.path.join(exp_path,'worker' + str(id)))
    commLogger = LearningLogger(path=os.path.join(exp_path,'worker' + str(id)), id='communication', level = 'INFO')
    comm.setLearningLogger(commLogger)
    w.setCommunicator(comm)
    logger = LearningLogger(path=exp_path, id='worker' + str(id), level = 'INFO')
    
    learner = learnerFactory.getLearner()
    learner.setLearningLogger(logger)
    
    stoppingCriterion = {stoppingCriterionName}({stoppingCriterionParams}) 
    learner.setStoppingCriterion(stoppingCriterion)
    w.setLearner(learner)
    print('created worker ' + nodeId + '\\n')
    w.run()
        

if __name__ == '__main__':
    learnerId = int(sys.argv[1])
    numberOfNodes = int(sys.argv[2])
    
    dataScheduler = {dataScheduler}()
    
    dataSourceFactory = {dataSourceFactoryName}({dataSourceFactoryParams})
    dataSource = dataSourceFactory.getDataSource(nodeId = learnerId)
    dataScheduler.setDataSource(source = dataSource)
    
    learnerFactory = {learnerFactoryName}({learnerFactoryParams})
    
    
    exp_path = '{exp_path}'
    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)
        
    commLogger = LearningLogger(path=exp_path, id='communication', level='NORMAL')
    Communicator.learningLogger = commLogger
    
    #t = Process(target = createWorker, args=(learnerId, exp_path, dataScheduler, learnerFactory, ), name = 'worker_' + str(learnerId))
    #t.daemon = True
    #t.start()
    createWorker(learnerId, exp_path, dataScheduler, learnerFactory, )
    while True:
        time.sleep(100)