#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import pandas
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from scipy.sparse import linalg as sparse_linalg

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

N = 10
J = 1
hx = 1
hz = 1
#init system
system = unlocking_System([0,1],"periodic",2,N)
system.gen_basis()
system_syms = model_sym_data(system,[translational(system)])

#create Hamiltonian
H = Hamiltonian(system,system_syms)
H.site_ops[1] = np.array([[0,1],[1,0]])
H.site_ops[2] = np.array([[-1,0],[0,1]])
H.model = np.array([[1,1],[2],[1]])
H.model_coef = np.array([J,hz,hx])

#dynamics following quench from |00000>
psi = ref_state(0,system)
k=system_syms.find_k_ref(psi.ref)
for n in range(0,np.size(k,axis=0)):
    H.gen(k[n])
    H.sector.find_eig(k[n])
    print(H.sector.eigvalues(k[n]))
    eig_overlap(psi,H,k[n]).plot()
plt.show()
fidelity(psi,H,"use sym").plot(np.arange(0,20,0.01),psi)
plt.show()
