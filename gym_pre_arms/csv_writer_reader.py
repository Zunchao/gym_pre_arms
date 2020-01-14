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

    def expand_data(self, x, n):
        """
        return array where each row value expand to n close values(within diameter of 0.01)
        [a] -> [a, a+<0.01, a->0.01, ...]
        :param x:
        :param n:
        :return:
        """
        a = x.shape[0]
        b = x.shape[1]
        xnew = np.empty([a, b*(n+1)])
        for i in range(a):
            xi = x[i]
            for j in range(b):
                y = np.random.rand(n)-0.5
                xi=np.append(xi, y/50+xi[j])
            xnew[i] = xi
        return xnew


if __name__ == '__main__':
    test = CSV_Writer_Reader()
    x = np.random.rand(10, 2)
    print(np.array(x))
    test.writecsv('/home/zheng/ws_xiao/gym_test/gym_pre_arms/test.csv', x)
    d = test.readcsv('/home/zheng/ws_xiao/gym_test/gym_pre_arms/X.csv')
    print(len(d), np.array(d))
    #d[1].append(1.0)
    d = np.array(d)
    dd = np.empty([9,6])
    for i in range(d.shape[0]):
        a = d[i]
        a1 = d[i]
        for j in range(d.shape[1]):
            x = np.random.rand(2)-0.5
            a=np.append(a, x/50+a[0:2])
        dd[i] = a
    ddd=test.expand_data(d,3)
    print(dd, ddd)