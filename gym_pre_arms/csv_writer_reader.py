import csv
import numpy as np

class CSV_Writer_Reader():
    def __init__(self):
        self._path = ''  #unused
        self._data = []  #unused

    def writecsv(self, filepath, data):
        with open(filepath, 'w+') as csvfile:
            csvfile.truncate()
            writer = csv.writer(csvfile, delimiter=' ')
            [writer.writerow(r) for r in data]

    def readcsv(self, filepath):
        i = 0
        data = []
        with open(filepath, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            read = [[float(e) for e in r] for r in reader]
        '''
        while True:
            try:
                data.append(read[i])
                i+=1
            except IndexError:
                break
        '''
        return read


if __name__ == '__main__':
    test = CSV_Writer_Reader()
    x = np.random.rand(10, 2)
    print(np.array(x))
    test.writecsv('/home/zheng/ws_xiao/gym_test/gym_pre_arms/test.csv', x)
    d = test.readcsv('/home/zheng/ws_xiao/gym_test/gym_pre_arms/X.csv')
    print(len(d), np.array(d))
