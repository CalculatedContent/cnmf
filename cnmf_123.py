#!/usr/bin/env python

from numpy import *
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.datasets import fetch_mldata

import logging
from time import time

from numpy.random import RandomState
import pylab as pl

from sklearn import decomposition
import pymf 

# Loading data
mnist = fetch_mldata("MNIST original", data_home="./MNIST")
X, Y = mnist.data, mnist.target

# 70000 x (28x28)
# 70000 x 784
print X.shape, Y.shape

# subselect each set of numbers

cnts = {0:0}
for idy in  range(10):
    cnts[idy] =0 

for idx in range(len(Y)):
    idy = int(Y[idx])
    cnts[idy] = cnts[idy] + 1

for idy in  range(10):
    print "cnts  ",idy,cnts[idy]

n_samples, n_features = X.shape
print n_samples, n_features

# select data set num 
num = 0
data = zeros(shape=(cnts[num],n_features))
idd = 0
for idx in range(len(Y)):
    idy = int(Y[idx])
    if idy==num:
        data[idd]= X[idx]
        idd+=1

print data.shape


n_row, n_col = 10,10
n_components = n_row * n_col


###############################################################################
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    pl.figure(figsize=(2. * n_col, 2.26 * n_row))
    pl.suptitle(title, size=16)
    for i, comp in enumerate(images):
        pl.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        pl.imshow(comp.reshape(image_shape), cmap=pl.cm.gray,
                  interpolation='nearest',
                  vmin=-vmax, vmax=vmax)
        pl.xticks(())
        pl.yticks(())
    pl.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

###############################################################################


###############################################################################
def print_gallery(images, n_col=n_col, n_row=n_row):
    for i, comp in enumerate(images):
        print i, comp.max(), comp.size
#        print comp.reshape(image_shape)

###############################################################################
image_shape = (28,28)
rng = RandomState(0)

# run nmf on data
#estimator =  decomposition.NMF(n_components=n_components, init='nndsvd', tol=5e-3)
#estimator.fit(data)
#components_ = estimator.components_
#plot_gallery('NMF init',components_ )
#pl.show()

# why is nndsvd missing and how do we fix?

nndsvd_mdl = pymf.NNDSVD(data, num_bases=n_components)
nndsvd_mdl.factorize()
plot_gallery('NNDSVD', nndsvd_mdl.H )
pl.show()


#kmeans_mdl = pymf.Kmeans(data, num_bases=n_components)
#kmeans_mdl.factorize()
#plot_gallery('Kmeans', kmeans_mdl.H )
#pl.show()



nmf_mdl = pymf.CNMF(data, num_bases=n_components)
nmf_mdl.factorize(niter=1000)

print nmf_mdl.frobenius_norm()
print "H",nmf_mdl.H.shape
print "W",nmf_mdl.W.shape


plot_gallery('CNMF seeded with kmeans' ,nmf_mdl.H)
pl.show()



