from DLplatform.dataprovisioning import DataSource

import numpy as np
from sklearn.datasets import load_svmlight_file

'''
Generic DataSource for reading examples from a file
The function that creates examples with labels is passed as a parameter

'''
class FileDataSource(DataSource):
    def __init__(self, filename, decodeLine, name, indices, nodeId, numberOfNodes, shuffle=False, cache=False):
        DataSource.__init__(self, name=name)

        self._filename = filename
        self._cache = cache
        if self._cache:
            self._cachedData = []
            for l in open(filename, "r").readlines():
                if len(l) > 2:
                    self._cachedData.append(l)
        else:
            self._inputFileHandle = open(filename, "r")
            self._lineNo = 0
            
        self._indices = getattr(self, indices)(nodeId, numberOfNodes)
        self._shuffle = shuffle
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._decode_example = decodeLine
        self._examplesCounter = -1
        self._usedExamplesCounter = 0

    '''
    When using multiprocessing, the data souce is serialized using pickle (in windows, not so under linux). 
    However, the file handle cannot be pickled, since it contains a thread.lock object.
    To avoid this, we implemented the following two functions which govern the behavior of pickle.
    In here, the file handle object is disregarded and reopened in the child process, later. 
    We only need to take care to jump to the right line, afterwards, although in practic, this should never happen.
    '''
    def __getstate__(self):
        d = self.__dict__.copy()
        if '_inputFileHandle' in d:
            d['_inputFileHandle'] = None
        return d
    
    def __setstate__(self, d):
        if '_inputFileHandle' in d and d['_inputFileHandle'] == None:
            d['_inputFileHandle'] = open(d['_filename'], "r")
            for _ in range(d['_lineNo']):
                next(d['_inputFileHandle'])
        self.__dict__.update(d)
        
    def readLine(self):
        current_line = "\n"
        try:
            current_line = self._inputFileHandle.__next__()
            self._lineNo += 1
        except StopIteration:
            current_line = "\n"
        if current_line == "\n":
            self._inputFileHandle.close()
            self._inputFileHandle = open(self._filename, "r") 
            current_line = self._inputFileHandle.__next__()
            self._lineNo = 0
            self._examplesCounter = -1
            self.checkEpochEnd()
        return current_line

    def getNext(self):
        if self._cache:
            current_line = self._cachedData[self._indices[self._usedExamplesCounter]]
        else:
            while not self._examplesCounter == self._indices[self._usedExamplesCounter]:
                current_line = self.readLine()
                self._examplesCounter += 1
        self._usedExamplesCounter += 1
        self.checkEpochEnd()
        example = self._decode_example(current_line)
        return example

    def checkEpochEnd(self):
        if self._usedExamplesCounter == len(self._indices):
            self._usedExamplesCounter = 0
            if self._shuffle:
                np.random.shuffle(self._indices)
                
    def roundRobin(self, nodeIndexNumber, numberOfNodes):
        indices = []
        counter = 0
        for l in open(self._filename, "r").readlines():
            if len(l) > 2 and counter % numberOfNodes == nodeIndexNumber:
                indices.append(counter)
            counter += 1
        return indices
    
    def parallelRun(self, nodeIndexNumber, numberOfNodes):
        indices = []
        counter = 0
        for l in open(self._filename, "r").readlines():
            if len(l) > 2:
                indices.append(counter)
            counter += 1
        return indices
    
#     def non_iid(self, nodeIndexNumber, numberOfNodes):
#         print(numberOfNodes)
#         print(nodeIndexNumber)
# 
#         if numberOfNodes == 1:
#             print("Right branch")
#             print(numberOfNodes)
#             print(nodeIndexNumber)
#             with open('/home/IAIS/jsicking/vw_collaborative_learning/noniid_experiments/actual_noniid_experiments/index_lists/indexList_pretraining_iid_0.3.pickle', 'rb') as fp:
#                 learner_index_lists = pickle.load(fp)
# 
#         elif numberOfNodes == 5:
# 
#             with open('/home/IAIS/jsicking/vw_collaborative_learning/noniid_experiments/actual_noniid_experiments/index_lists/indexLists_afterPretraining_5_EMD=0.55.pickle', 'rb') as fp:
#                 learner_index_lists = pickle.load(fp)
# 
#         return learner_index_lists[nodeIndexNumber]

class SVMLightDataSource(FileDataSource):
    def __init__(self, filename, indices, name, nodeId, numberOfNodes, shuffle=False):
        DataSource.__init__(self, name=name)

        self._filename = filename
        self._indices = getattr(self, indices)(nodeId, numberOfNodes)
        self._shuffle = shuffle
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._examplesCount = 0
        X, y = load_svmlight_file(filename)
        self.X = np.array(X.todense())
        self.y = y

    ''' 
    return: (instance, label) with instance being a numpy array and label being a float
    '''
    def getNext(self):
        instance = self.X[self._indices[self._examplesCount]]
        label = self.y[self._indices[self._examplesCount]]
        self._examplesCount += 1
        self.checkEpochEnd()
        return instance, label

    def checkEpochEnd(self):
        if self._examplesCount == len(self._indices):
            self._examplesCount = 0
            if self._shuffle:
                np.random.shuffle(self._indices)