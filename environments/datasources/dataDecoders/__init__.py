import numpy as np

class DataDecoder:
    def __init__(self):
        pass
    
    def __call__(self, line):
        return ([],0)
    
class CSVDecoder(DataDecoder):
    def __init__(self, delimiter = ',', labelCol = -1):
        self._delimiter = delimiter
        self._labelCol = labelCol
        
    def __call__(self, line):
        parsed_line = [float(c) for c in line.split(self._delimiter)]
        label = parsed_line[self._labelCol]
        del parsed_line[self._labelCol]
        example = parsed_line
        
        return np.array(example), int(label)

    def __str__(self):
        return "CSV decoder"
