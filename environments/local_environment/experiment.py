from multiprocessing import Process
from DLplatform.coordinator import Coordinator, InitializationHandler
from DLplatform.worker import Worker
from DLplatform.communicating import Communicator, RabbitMQComm
from DLplatform.dataprovisioning import IntervalDataScheduler
from DLplatform.learningLogger import LearningLogger
from DLplatform.learning import LearnerFactory
import time
import os
import pickle
import numpy as np

class Experiment():    
    def __init__(self, messengerHost, messengerPort, numberOfNodes, sync, aggregator, learnerFactory, dataSourceFactory, stoppingCriterion, initHandler = InitializationHandler(), sleepTime = 5):
        self.messengerHost = messengerHost
        self.messengerPort = messengerPort
        self.numberOfNodes = numberOfNodes
        self.sync = sync
        self.aggregator = aggregator
        self.learnerFactory = learnerFactory
        self.dataSourceFactory = dataSourceFactory
        self.stoppingCriterion = stoppingCriterion
        self.initHandler = initHandler
        self._uniqueId = str(os.getpid())
        self.sleepTime = sleepTime

    def run(self, name):
        self.start_time = time.time()
        exp_path = name + "_" + self.getTimestamp()
        os.mkdir(exp_path)
        self.writeExperimentSummary(exp_path, name)
        t = Process(target = self.createCoordinator, args=(exp_path, ), name = 'coordinator')    
        #t.daemon = True
        t.start()
        time.sleep(self.sleepTime)
        for id in range(self.numberOfNodes):
            t = Process(target = self.createWorker, args=(id, exp_path, ), name = "worker_" + str(id))
            #t.daemon = True
            t.start()
            time.sleep(self.sleepTime)
        os.waitpid(-1, 0)

    def createCoordinator(self, exp_path):
        coordinator = Coordinator()
        coordinator.setInitHandler(self.initHandler)
        comm = RabbitMQComm(hostname = self.messengerHost, port = self.messengerPort, user = 'guest', password = 'guest', uniqueId = self._uniqueId)
        os.mkdir(os.path.join(exp_path,'coordinator'))
        commLogger = LearningLogger(path=os.path.join(exp_path,'coordinator'), id="communication", level = 'INFO')
        comm.setLearningLogger(commLogger)
        coordinator.setCommunicator(comm)
        self.sync.setAggregator(self.aggregator)
        coordinator.setSynchronizer(self.sync)
        logger = LearningLogger(path=exp_path, id="coordinator", level = 'INFO')
        coordinator.setLearningLogger(logger)
        print("Starting coordinator...\n")
        coordinator.run()        

    def createWorker(self, id, exp_path):
        print("start creating worker" + str(id))
        nodeId = str(id)
        w = Worker(nodeId)
        dataScheduler = IntervalDataScheduler()
        dataSource = self.dataSourceFactory.getDataSource(nodeId = id)
        dataScheduler.setDataSource(source = dataSource)
        w.setDataScheduler(dataScheduler)
        comm = RabbitMQComm(hostname = self.messengerHost, port = self.messengerPort, user = 'guest', password = 'guest', uniqueId = self._uniqueId)
        os.mkdir(os.path.join(exp_path,"worker" + str(id)))
        commLogger = LearningLogger(path=os.path.join(exp_path,"worker" + str(id)), id="communication", level = 'INFO')
        comm.setLearningLogger(commLogger)
        w.setCommunicator(comm)
        logger = LearningLogger(path=exp_path, id="worker" + str(id), level = 'INFO')
        # TODO: we want to make it cleaner, without a string for eval
        learner = self.learnerFactory.getLearner()
        learner.setLearningLogger(logger)
        learner.setStoppingCriterion(self.stoppingCriterion)
        w.setLearner(learner)
        print("create worker " + nodeId + "\n")
        w.run()
    
    def getTimestamp(self):
        return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    
    def writeExperimentSummary(self, path, name):
        outString = "Experiment " + name + " Summary:\n\n"
        outString += "Start:\t" + str(self.start_time) + "\n"
        outString += "Number of Nodes:\t"+str(self.numberOfNodes)+"\n"
        outString += "Learner:\t\t\t"+str(self.learnerFactory)+"\n"
        outString += "Data source:\t\t"+str(self.dataSourceFactory)+"\n"
        outString += "Sync:\t\t\t"+str(self.sync)+"\n"
        outString += "Aggregator:\t\t"+str(self.aggregator)+"\n"
        outString += "Stopping criterion:\t"+str(self.stoppingCriterion)+"\n"
        outString += "Messenger Host:\t\t"+str(self.messengerHost)+"\n"
        outString += "Messenger Port:\t\t"+str(self.messengerPort)+"\n"
        
        summaryFile = os.path.join(path, "summary.txt")
        f = open(summaryFile, 'w')
        f.write(outString)
        f.close()
