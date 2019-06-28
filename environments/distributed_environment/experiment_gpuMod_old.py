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

class Experiment_gpuMod():    
    def __init__(self, messengerHost, messengerPort, messengerUser, messengerPassword, sync, aggregator, learnerCreator, dataSourceFunction, indicesFunction, shuffleData, cacheData, stoppingCriterion):
        self.messengerHost = messengerHost
        self.messengerPort = messengerPort
        self.messengerUser = messengerUser
        self.messengerPassword = messengerPassword
        self.sync = sync
        self.aggregator = aggregator
        self.learnerCreator = learnerCreator
        self.dataSourceFunction = dataSourceFunction
        self.indicesFunction = indicesFunction
        self.shuffleData = shuffleData
        self.cacheData = cacheData
        self.stoppingCriterion = stoppingCriterion

    def run(self, name, username, nodes, gpu_numbers):
        self.runCoordinator(name, username, nodes, gpu_numbers)

    def runCoordinator(self, name, username, nodes, gpu_numbers):
        localClusterPath = "/data/user/" + username + "/vwplatform/experiments/distributed_experiments"
        exp_path = os.path.join(localClusterPath, name + "_" + self.getTimestamp())
        os.mkdir(exp_path)
        t = Process(target = self.createCoordinator, args=(exp_path, ), name = 'coordinator')    
        #t.daemon = True
        t.start()
        time.sleep(5)
        numberOfNodes = len(nodes)
        for i in range(numberOfNodes):
            # TODO later want also pass the id of GPU to use; will require passing it as a parameter further on to the learner in DLPlatform
            sys_path = '/data/user/' + username + '/vwplatform/experiments'
            print("TEEEEEEEEEEEST"+str(gpu_numbers[i])) 
            nodeScript = self.getNodeScript(sys_path, exp_path, username, self.dataSourceFunction, self.indicesFunction, gpu_numbers[i])
            node_script_file = os.path.join(sys_path, 'distributed_experiments', name + "_node_script.py")
            subprocess.Popen("ssh " + username + "@" + nodes[i] + " \"echo \\\"" + nodeScript + "\\\" > " + node_script_file + " && source ~/.remote_ssh && python " + node_script_file + " " + str(i) + " " + str(numberOfNodes) + "\"", shell = True)
        while True:
            time.sleep(100)

    def createCoordinator(self, exp_path):
        coordinator = Coordinator()
        learner = eval(self.learnerCreator.replace('\\', ''))
        initialParams = learner.getParameters()
        coordinator.setInitialParams(initialParams)
        comm = RabbitMQComm(hostname = self.messengerHost, port = self.messengerPort, user = self.messengerUser, password = self.messengerPassword)
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

    def getNodeScript(self, sys_path, exp_path, username, dataSourceFunction, indicesFunction, gpu_number):
        nodeScript = '''\
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']='{gpu_number}'
sys.path.append('{sys_path}')
from local_environment import DataSourceFactory
from distributed_environment import Experiment
from DLplatform.dataprovisioning import MNISTDataSource
from DLplatform.synchronizing import PeriodicSync
from DLplatform.learning import LearnerFactory
from DLplatform.aggregating import Average
from DLplatform.worker import Worker
from DLplatform.communicating import Communicator, RabbitMQComm
from DLplatform.dataprovisioning import BasicDataScheduler
from DLplatform.learningLogger import LearningLogger
from DLplatform.learning import LearnerFactory
from DLplatform.learningLogger import LearningLogger
from multiprocessing import Process
import time

def createWorker(id, exp_path, dataSource):
	nodeId = str(id)
	w = Worker(nodeId)
	dataScheduler = BasicDataScheduler()
	dataScheduler.setDataSource(source = dataSource)
	w.setDataScheduler(dataScheduler)
	comm = RabbitMQComm(hostname ='{mHost}', port = {mPort}, user = '{mUser}', password = '{mPwd}')
	os.mkdir(os.path.join(exp_path,'worker' + str(id)))
	commLogger = LearningLogger(path=os.path.join(exp_path,'worker' + str(id)), id='communication')
	comm.setLearningLogger(commLogger)
	w.setCommunicator(comm)
	logger = LearningLogger(path=exp_path, id='worker' + str(id), level='NORMAL')
	learner = eval('{lCreator}')
	learner.setLearningLogger(logger)
	learner.setStoppingCriterion({sCriterion})
	w.setLearner(learner)
	print('create worker ' + nodeId)
	w.run()

if __name__ == '__main__':
	username = '{username}'
	id = int(sys.argv[1])
	numberOfNodes = int(sys.argv[2])
	dsFactory = DataSourceFactory()
	sourceGetter = getattr(dsFactory, '{dsFunc}')
	dataSource = sourceGetter(nodeIndexNumber = id, numberOfNodes = numberOfNodes, indices = '{indFunc}', shuffle = '{shuffle}', cache = '{cache}')
	exp_path = '{exp_path}'
	if not os.path.isdir(exp_path):
		os.mkdir(exp_path)
	commLogger = LearningLogger(path=exp_path, id='communication', level='NORMAL')
	Communicator.learningLogger = commLogger
	t = Process(target = createWorker, args=(id, exp_path, dataSource, ), name = 'worker_' + str(id))
	#t.daemon = True
	t.start()
	while True:
		time.sleep(100)
        '''.format(sys_path = sys_path, mHost = self.messengerHost, mPort = self.messengerPort, mUser = self.messengerUser, mPwd = self.messengerPassword, lCreator = self.learnerCreator, username = username, dsFunc = dataSourceFunction, indFunc = indicesFunction, shuffle = self.shuffleData, cache = self.cacheData, sCriterion = self.stoppingCriterion, exp_path = exp_path, gpu_number = gpu_number)
        return nodeScript
        
