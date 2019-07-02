from multiprocessing import Process
from DLplatform.coordinator import Coordinator, InitializationHandler
from DLplatform.worker import Worker
from DLplatform.communicating import Communicator, RabbitMQComm
from DLplatform.dataprovisioning import IntervalDataScheduler
from DLplatform.learningLogger import LearningLogger
from DLplatform.learning import LearnerFactory

import time
import os
import subprocess

class Experiment():    
    def __init__(self, expFileName, sysPath, messengerHost, messengerPort, messengerUser, messengerPassword, sync, aggregator, 
                 learnerFactory, dataSourceFactory, stoppingCriterion, initHandler = InitializationHandler()):
        self._uniqueId      = str(os.getpid())
        self.expFileName    = expFileName
        self.sysPath        = sysPath
        
        self.messengerHost     = messengerHost
        self.messengerPort     = messengerPort
        self.messengerUser     = messengerUser
        self.messengerPassword = messengerPassword
        
        self.sync = sync
        self.aggregator = aggregator
        self.learnerFactory = learnerFactory
        self.dataSourceFactory = dataSourceFactory
        self.dataScheduler = IntervalDataScheduler()
    
        self.initHandler = initHandler
        self.stoppingCriterion = stoppingCriterion
        

    def run(self, name, username, nodes):
        self.runCoordinator(name, username, nodes)

    def runCoordinator(self, name, username, nodes):
        localClusterPath = os.path.dirname(self.expFileName)#self.sysPath + "experiments/experiments/distributed_experiments"
        exp_path = os.path.join(localClusterPath, name + "_" + self.getTimestamp())
        app_path = os.path.dirname(self.expFileName).split("dlapplication-dev")[0] + "dlapplication-dev/"
        os.mkdir(exp_path)
        t = Process(target = self.createCoordinator, args=(exp_path, ), name = 'coordinator')    
        #t.daemon = True
        t.start()
        time.sleep(5)
        numberOfNodes = len(nodes)
        for i in range(numberOfNodes):
            # TODO later want also pass the id of GPU to use; will require passing it as a parameter further on to the learner in DLPlatform                    
            nodeScript = self.getNodeScript(self.sysPath, app_path, exp_path)
            nodeScriptDirectory = os.path.dirname(self.expFileName)
            node_script_file = os.path.join(nodeScriptDirectory, name + "worker_"+str(i)+"_node_script.py")
            subprocess.Popen("ssh " + username + "@" + nodes[i] + " \"echo \\\"" + nodeScript + "\\\" > " + node_script_file + " && source ~/.remote_ssh && python " + node_script_file + " " + str(i) + " " + str(numberOfNodes) + "\"", shell = True)
        os.waitpid(-1,0)

    def createCoordinator(self, exp_path):
        coordinator = Coordinator()
        coordinator.setInitHandler(self.initHandler)
        comm = RabbitMQComm(hostname = self.messengerHost, port = self.messengerPort, user = self.messengerUser, password = self.messengerPassword, uniqueId = self._uniqueId)
        os.mkdir(os.path.join(exp_path,'coordinator'))
        commLogger = LearningLogger(path=os.path.join(exp_path,'coordinator'), id="communication")
        comm.setLearningLogger(commLogger)
        coordinator.setCommunicator(comm)
        self.sync.setAggregator(self.aggregator)
        coordinator.setSynchronizer(self.sync)
        logger = LearningLogger(path=exp_path, id="coordinator", level='NORMAL')
        coordinator.setLearningLogger(logger)
        print("Starting coordinator...\n")
        coordinator.run()   
    
    def getTimestamp(self):
        return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))

    def getNodeScript(self, sys_path, app_path, exp_path):
        imports = ""
        for line in open(self.expFileName, 'r').readlines():
            if "import" in line:
                imports += line
        
        
        templatePath = app_path + "environments/distributed_environment" 
        #print(templatePath)
        scriptTemplate = open(os.path.join(templatePath,'nodeScriptTemplate.py'), 'r').read()
        nodeScript = scriptTemplate.format(dlplatform_path = sys_path, dlapplication_path = app_path, exp_path = exp_path, imports = imports, 
                                           mHost = self.messengerHost, mPort = self.messengerPort, 
                                           mUser = self.messengerUser,  mPwd = self.messengerPassword, 
                                           uniqueId = self._uniqueId,  
                                           learnerFactoryName = self.learnerFactory.__class__.__name__,
                                           learnerFactoryParams = self.learnerFactory.getInitParameters(),
                                           dataSourceFactoryName = self.dataSourceFactory.__class__.__name__,
                                           dataSourceFactoryParams = self.dataSourceFactory.getInitParameters(),
                                           stoppingCriterionName = self.stoppingCriterion.__class__.__name__,
                                           stoppingCriterionParams = self.stoppingCriterion.getInitParameters(),
                                           dataScheduler = self.dataScheduler.__class__.__name__)
        return nodeScript
