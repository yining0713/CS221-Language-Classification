import sys
import os


def numpy_array_to_string(np_array):
    return np.array2string(np_array, precision=8, separator=',').replace('\n', '')[1:-1]

class Log:
    def __init__(self, log="../log/run_all.log"):
        script_dir = os.path.split(os.path.realpath(sys.argv[0]))[0]
        logfile = os.path.join(script_dir, log)
        self.logwriter = open(logfile, 'a+')
    
    def writelog(self, message: str):
        self.logwriter.write(message + '\n')

    def write_numpy_array(self, np_array):
        if type(np_array.size) == int or len(np_array.size) == 1:
            for x in np_array:
                self.writelog(str(x))
        elif len(np_array.size) == 2:
            for x in np_array:
                y = numpy_array_to_string(x)
                self.writelog(y)
        else:
            self.writelog("array with more than 2 dimensions")
    
    def write_newline(self):
        self.writelog('\n')
        


logging = Log()
