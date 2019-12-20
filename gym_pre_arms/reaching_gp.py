import GPy
from IPython.display import display
import numpy as np

class ReachingGPy():
    def __init__(self):
        self.path = ''

    def gp_process(self, X, Y):
        kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
        m = GPy.models.GPRegression(X, Y, kernel)
        m.optimize(messages=True)
        display(m)
        np.save('model_save.npy', m.param_array)

    def gp_load(self, X, Y):
        m_load = GPy.models.GPRegression(X, Y, initialize=False)
        m_load.update_model(False)  # do not call the underlying expensive algebra on load
        m_load.initialize_parameter()  # Initialize the parameters (connect the parameters up)
        m_load[:] = np.load('model_save.npy')  # Load the parameters
        m_load.update_model(True)  # Call the algebra only once
        #print(m_load)
        display(m_load)