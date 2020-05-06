import GPy
from IPython.display import display
import numpy as np
from csv_writer_reader import CSV_Writer_Reader
import os
import inspect
import pickle
import time
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

class ReachingGPy():
    def __init__(self):
        self.path = ''

    def gp_process(self, X, Y):
        kernel = GPy.kern.RBF(input_dim=4, variance=1., lengthscale=1.)
        m = GPy.models.SparseGPRegression(X, Y, kernel)
        m.optimize(messages=True)
        display(m)
        np.save('model_save11.npy', m.param_array)

        with open(currentdir+'/model_save12.dump', 'wb') as buff:
            pickle.dump(m, buff)

    def gp_load(self, X, Y, x0):
        m_load = GPy.models.GPRegression(X, Y, initialize=False)
        m_load.update_model(False)  # do not call the underlying expensive algebra on load
        m_load.initialize_parameter()  # Initialize the parameters (connect the parameters up)
        m_load[:] = np.load('model_save.npy')  # Load the parameters
        m_load.update_model(True)  # Call the algebra only once
        #print(m_load)
        display(m_load)
        y, ysigma = m_load.predict(Xnew=x0)
        print('final 1 : ', y, ysigma)

        m_load = pickle.load(open(currentdir+'/model_save12.dump', "rb"))
        y, ysigma = m_load.predict(Xnew=x0)
        print('final 2 : ', y, ysigma)


if __name__ == '__main__':
    start_time = time.time()
    pathin = currentdir+'/inputX_ori_rand.csv'
    pathout = currentdir+'/outputY_ori_rand.csv'
    print(pathin)
    # split into input (X) and output (Y) variables
    csvrw = CSV_Writer_Reader()
    X = csvrw.readcsv(filepath=pathin)
    Y = csvrw.readcsv(filepath=pathout)

    x = np.array(X[0:9999])
    y = np.array(Y[0:9999])

    x1 = np.array(X[0:1])
    y1 = np.array(Y[0:1])
    x0 = np.array(X[99951:99954])

    test = ReachingGPy()
    test.gp_process(x, y)
    for i in range(x0.shape[0]):
        print('input : ', np.array([x0[i]]))
        test.gp_load(x1, y1, np.array([x0[i]]))
    runningtime = time.time() - start_time
    print("--- %s seconds ---" % (time.time() - start_time))