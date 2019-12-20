import csv
import numpy

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
        with open(filepath, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            data = [[float(e) for e in r] for r in reader]
        return data


if __name__ == '__main__':
    test = CSV_Writer_Reader()
    x = numpy.random.rand(10,2)
    print(numpy.array(x))
    test.writecsv('/home/zheng/ws_xiao/gym_test/gym_pre_arms/test.csv', x)
    d = test.readcsv('/home/zheng/ws_xiao/gym_test/gym_pre_arms/Y.csv')
    print(d)
