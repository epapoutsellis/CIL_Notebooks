# -*- coding: utf-8 -*-
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy
from cil.optimisation.functions import Function

class SmoothKullbackLeibler(Function):
        
    """The smooth version of the Kullback-Leibler divergence functional.

    References
    ----------
    [CERS2017] A. Chambolle, M. J. Ehrhardt, P. Richtarik and C.-B. Schoenlieb,
    *Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling
    and Imaging Applications*. ArXiv: http://arxiv.org/abs/1706.04957 (2017).
    """    
      
    
    def __init__(self, **kwargs):
                
        self.b = kwargs.get('b', None)
        self.eta = kwargs.get('eta',None)        
        
        if self.b is None :
            raise ValueError('Please define data, as b = ...')

        if (self.b.as_array() < 0).any():            
            raise ValueError('Data should be positive')             
            
        if self.eta is None:
            raise ValueError('Please define data, as eta = ...')
                        
        if (self.eta.as_array() <= 0).any():            
            raise ValueError('Background should be positive')                          
                
        super(SmoothKullbackLeibler, self).__init__(L = numpy.max(self.b / self.eta ** 2))
                          
    
    def __call__(self, x):
                
        r"""Returns the value of the smooth KullbackLeibler functional at :math:`x`.
        """
                                
#         out = self.b*0.
        
        out = numpy.zeros(shape=self.b.shape, dtype=numpy.float32)
        tmp_x = x.as_array()
        tmp_b = self.b.as_array()
        tmp_eta = self.eta.as_array()
    
#         i = x>=0
        i = x.as_array()
#         out.array[i] = x.array[i] + self.eta.array[i] - self.b.array[i]    
        out[i] = x[i] + self.eta[i] - self.b.array[i]
        
        j = self.b>0
        k = numpy.logical_and(i, j) 
        out.array[k] += self.b.array[k] * numpy.log((self.b.array[k] / (x.array[k] + self.eta.array[k]))) 
                
        i = numpy.logical_not(i)
        out.array[i] += (self.b.array[i] / (2 * self.eta.array[i]**2) * x.array[i]**2 + (1 - self.b.array[i] / self.eta.array[i]) * x.array[i] +
                   self.eta.array[i] - self.b.array[i])

        k = numpy.logical_and(i,j)
        out.array[k] += self.b.array[k] * numpy.log(self.b.array[k] / self.eta.array[k])
        
        return out.sum()         
    
    def gradient(self, x, out=None):
        
        r"""Returns the value of the gradient of the smooth KullbackLeibler functional at :math:`x`.
        """        
        
        i = x.as_array()>=0
        j = x.as_array()<0
        
        tmp = x
        tmp = x*0.
        
        
#         i = x>=0
#         j = x<0
#         tmp = x*0.
        
        tmp[i]
        
        tmp.array[i] = 1. - self.b.array[i]/(x.array[i] + self.eta.array[i])
        tmp.array[j] = self.b.array[j]*x.array[j]/self.eta.array[j]**2 + 1. - self.b.array[j]/(self.eta.array[j])
        
        if out is None:
            return tmp
        else:
            out.fill(tmp)
            
    def convex_conjugate(self, x):
        
        r"""Returns the value of the convex conjugate of the smooth KullbackLeibler functional at :math:`x`.
        """      
        # this is
#         if (x>=1).sum()>0:
#             return numpy.inf
        
        tmp = x * 0.
        
        i = x<(1 - self.b/self.eta)
        ry = self.eta.array[i]**2 / self.b.array[i]
                
        tmp.array[i] += (ry/2 * x.array[i]**2 + (self.eta.array[i] - ry) * x.array[i] + ry/2 + 3/2 * self.b.array[i] - 2 * self.eta.array[i]) 
        
        j = self.b>0
        k = numpy.logical_and(i,j)
        tmp.array[k] -= self.b.array[k]*numpy.log(self.b.array[k]/self.eta.array[k])
                
        i = numpy.logical_not(i)
        tmp.array[i] -= self.eta.array[i]*x.array[i]
        
        k = numpy.logical_and(i, j)
        tmp.array[k] -= self.b.array[k]*numpy.log(1 - x.array[k])
                        
        return tmp.sum()

                                                                            
    def proximal_conjugate(self, x, tau, out = None):
        
        r"""Returns the value of the proximal conjugate of the smooth KullbackLeibler functional at :math:`x`.
        """        
        
        tmp = x*0.
        i = x<(1-self.b/self.eta)
        
        tmp.array[i] += (self.b.array[i]*x.array[i] - tau*self.eta.array[i]*self.b.array[i] + tau*self.eta.array[i]**2)/(self.b.array[i]+tau*self.eta.array[i]**2)
        
        i = numpy.logical_not(i)
        
        tmp.array[i] = x.array[i] + tau*self.eta.array[i] + 1 \
                - numpy.sqrt( (x.array[i] + tau*self.eta.array[i] - 1)**2 + 4*tau*self.b.array[i])
        tmp.array[i] *= 0.5
        
        if out is None:
            return tmp
        else:
            out.fill(tmp)
                              
                    
    def proximal(self, x, tau, out=None):
                
        raise NotImplementedError('Not Implemented')
                