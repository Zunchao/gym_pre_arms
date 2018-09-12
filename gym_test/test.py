import numpy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import tensorflow as tf
from scipy.stats import multivariate_normal, rv_continuous
import ouNoise


print('hgdh,',numpy.random.rand(2))


print(ouNoise.OUNoise(1))
print(numpy.__path__)
a = numpy.array([0,1,3,6,5,7])
print(a[-2:])
print(a[1:3])
assert 2!=1,'2不等于1'
from math import pi, sin, cos, tanh
def funA(a):
    print('funA')

def funB(b):
    print('funB')

@funA
@funB
def funC():
    print('funC')
#funC()

o= [1,1,1]
print(o[1]-1)
numpy.random.seed(12)
print('randomseed diff : ', numpy.random.rand(3))

print('random', numpy.random.choice(1000,10))

Mu_w = numpy.zeros(4)
Sigma_w = numpy.eye(4) * 1e6
sample = numpy.random.multivariate_normal(Mu_w, Sigma_w, 3)
z = multivariate_normal.pdf(sample, Mu_w, Sigma_w)
#sample[:] = 0
print('sample',sample, z)


means = numpy.array([0, 0, 0])
covs = numpy.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
#samples1 = numpy.random.multivariate_normal(means,covs)
#samples2 = numpy.random.multivariate_normal(means,covs)
#samples3 = numpy.random.multivariate_normal(means,covs)
#samples = [samples1, samples2, samples3]
# sample is a point of n-dimemsions same with the size of means

samples = numpy.random.multivariate_normal(means, covs, 3)
y = multivariate_normal.pdf(samples, means, covs); y
cy = numpy.random.choice([-1,1], 3)
yy = y*cy
rv = multivariate_normal(means, covs)
print('pdf', samples, y, cy, yy, numpy.log(samples))

print()
# ones(3) = [1, 1, 1]

mean = [0, 0]
cov = [[1, 0], [0, 1]]  # diagonal covariance
x, y = numpy.random.multivariate_normal(mean, cov, 500).T
#plt.plot(x/3, y/3, 'x')
#plt.axis('equal')
#plt.show()

x = numpy.linspace(0, 5, 10, endpoint=False)
y = multivariate_normal.pdf(x, mean=2.5, cov=0.5); y
print(y)

Mu_w = numpy.zeros(4)
Sigma_w = numpy.eye(4)*1e-1
samples = numpy.random.multivariate_normal(Mu_w, Sigma_w, 3)
dsamples = multivariate_normal.pdf(samples, Mu_w, Sigma_w)
#asamples = rv_continuous.cdf([1])
print('last:', samples, dsamples)

REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]
print([
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0] )

init_w = tf.random_normal_initializer(0., 0.3)
init_b = tf.constant_initializer(0.1)
print(init_b,init_w)