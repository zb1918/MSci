# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 12:17:16 2022

@author: zaza
"""

import numpy as np
import scipy as sc
from matplotlib import pyplot as plt


#---------------------------constants in cgs----------------------------------#
H_molecule = 1.00784        # mass of H in a.m.u.
amu = 1.66e-24              # 1 a.m.u. in g
H_mu = H_molecule * amu     # mass of a single H molecule in g
gamma = 5/3                 # 1 + 2/d.o.f. where d.o.f. of H is taken to be 3
kb = 1.38e-16               # boltzmann constant in cgs

#------------------------calculating temperature------------------------------#
T = U * H_mu * (gamma - 1)/(D * kb)
T = T.T