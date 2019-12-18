import GPy
import numpy as np
from IPython.display import display

X = np.random.uniform(-3.,3.,(20,3))
print(X, X.shape)
Y = np.sin(X[:,0:2]) + np.random.randn(20,2)*0.05
print(Y)
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

m = GPy.models.GPRegression(X,Y,kernel)

display(m)
m.optimize(messages=True)

x_test = np.random.uniform(-3.,3.,(1,2))
print(x_test)
y_test = m.predict(Xnew=x_test)
print(y_test)