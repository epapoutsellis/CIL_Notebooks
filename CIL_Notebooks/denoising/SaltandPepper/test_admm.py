# import libraries
import matplotlib.pyplot as plt
import numpy as np
from cil.optimisation.functions import L1Norm, MixedL21Norm, BlockFunction, ZeroFunction
from cil.optimisation.operators import BlockOperator, GradientOperator, IdentityOperator
from cil.optimisation.algorithms import LADMM, PDHG
from cil.framework import ImageGeometry
from cil.utilities.dataexample import TestData
import os, sys

# Load an image from the CIL gallery. 
loader = TestData(data_dir=os.path.join(sys.prefix, 'share','cil'))
data = loader.load(TestData.SHAPES)
ig = data.geometry

# Add gaussian noise
n1 = TestData.random_noise(data, mode = 's&p', amount=0.1, seed = 10)
noisy_data = data.geometry.allocate()
noisy_data.fill(n1)

alpha = 0.8

K = BlockOperator(GradientOperator(ig), IdentityOperator(ig))
G = BlockFunction(alpha * MixedL21Norm(), L1Norm(b=noisy_data))
F = ZeroFunction()

# Compute operator Norm
normK = K.norm()

# Primal & dual stepsizes 
sigma = 5.
tau = 1. / normK ** 2 

# Setup and run the PDHG algorithm
admm = LADMM(f=F, g=G, operator=K, max_iteration=500)
admm.run(100)