#!/usr/bin/env python
import pymf
import numpy as np
data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
nmf_mdl = pymf.NMF(data, num_bases=2)
nmf_mdl.factorize(niter=100)

print '2 instances, 3 features, 2 factors'

# reconstruction error, measured as 2-norm

print nmf_mdl.frobenius_norm()
print "H",nmf_mdl.H.shape
print "W",nmf_mdl.W.shape
