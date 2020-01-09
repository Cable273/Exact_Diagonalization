#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from progressbar import ProgressBar

import sys
file_dir = '/home/kieran/Desktop/Work/CMP/physics_code/Exact_Diagonalization/functions'
sys.path.append(file_dir)
file_dir = '/home/kieran/Desktop/Work/CMP/physics_code/Exact_Diagonalization/Classes'
sys.path.append(file_dir)
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m
from Search_functions import find_index_bisection
from State_Classes import sym_state,ref_state,zm_state

class eig_overlap:
    def bulk_eval(state,H,k_vec=None):
        if k_vec is None: #no symmetry, use full basis
            z_ref = bin_to_int_base_m(state,H.system.base)
            z_index = find_index_bisection(z_ref,H.system.basis_refs)
            z_energy = H.sector.eigvectors()[z_index,:]
            eigenvalues = H.sector.eigvalues()
        else: #symmetry used, just plot the given sym_block
            z_ref = bin_to_int_base_m(state,H.system.base)
            psi = ref_state(z_ref,H.system)
            z_mom = psi.sym_basis(k_vec,H.syms)
            # z_mom = z_mom * np.power(np.vdot(z_mom,z_mom),-0.5)
            z_energy = np.zeros(np.size(H.sector.eigvalues(k_vec)),dtype=complex)
            for n in range(0,np.size(z_energy,axis=0)):
                z_energy[n] = np.vdot(z_mom,H.sector.eigvectors(k_vec)[:,n])
            eigenvalues = H.sector.eigvalues(k_vec)

        overlap = np.log10(np.abs(z_energy)**2)
        return overlap

    def plot(state,H,k_vec=None,label=None):
        print("Plotting eigenstate overlap with given state...")
        if k_vec is None: #no symmetry, use full basis
            z_ref = bin_to_int_base_m(state,H.system.base)
            z_index = find_index_bisection(z_ref,H.system.basis_refs)
            z_energy = H.sector.eigvectors()[z_index,:]
            eigenvalues = H.sector.eigvalues()
        else: #symmetry used, just plot the given sym_block
            z_ref = bin_to_int_base_m(state,H.system.base)
            psi = ref_state(z_ref,H.system)
            z_mom = psi.sym_basis(k_vec,H.syms)
            # z_mom = z_mom * np.power(np.vdot(z_mom,z_mom),-0.5)
            z_energy = np.zeros(np.size(H.sector.eigvalues(k_vec)),dtype=complex)
            for n in range(0,np.size(z_energy,axis=0)):
                z_energy[n] = np.vdot(z_mom,H.sector.eigvectors(k_vec)[:,n])
            eigenvalues = H.sector.eigvalues(k_vec)

        overlap = np.log10(np.abs(z_energy)**2)
        to_del=[]
        for n in range(0,np.size(overlap,axis=0)):
            if overlap[n] <-10:
                to_del = np.append(to_del,n)
        for n in range(np.size(to_del,axis=0)-1,-1,-1):
            overlap=np.delete(overlap,to_del[n])
            eigenvalues=np.delete(eigenvalues,to_del[n])
            
        if label is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if k_vec is None:
            # plt.scatter(eigenvalues,overlap,alpha=0.6)
            from scipy.stats import gaussian_kde
            x = eigenvalues
            y = overlap
            plt.scatter(x,y)
             
            # Calculate the point density
            # xy = np.vstack([x,y])
            # z = gaussian_kde(xy)(xy)
             
            # # Sort the points by density, so that the densest points are plotted last
            # idx = z.argsort()
            # x, y, z = x[idx], y[idx], z[idx]
             
            # fig, ax = plt.subplots()
            # ax.scatter(x, y, c=z, s=50, edgecolor='')
        else:
            # plt.scatter(eigenvalues,overlap,label=str(k_vec)+r" $\pi /$"+str(H.system.N)+" Symmetry sector")
            plt.scatter(eigenvalues,overlap,color='blue')

        if label is not None:
            A=eigenvalues
            B=overlap
            for i,j in zip(A,B):
                # ax.annotate('%s)' %j, xy=(i,j), xytext=(30,0), textcoords='offset points')
                ax.annotate('(%s,' %i, xy=(i,j))

        plt.legend()
        plt.xlabel("E")
        plt.ylabel(r"$\vert \langle \psi_E \vert \psi \rangle \vert^2$")
        plt.title(str(H.system.base)+" Colour, N="+str(H.system.N))
        plt.legend()

def time_evolve_state(state_energy_basis,eigenvalues,t):
    phases = np.exp(-1j * t * eigenvalues)
    return np.multiply(state_energy_basis,phases)

class fidelity():
    def eval(state,t,H,use_syms=None):
        if use_syms==None:
            z_ref = bin_to_int_base_m(state,H.system.base)
            z_index = find_index_bisection(z_ref,H.system.basis_refs)
            z_init = H.sector.eigvectors()[z_index,:]
            eig_f = H.sector.eigvalues()
        else:
            z_init,eig_f = energy_basis(state,H)

        evolved_state = time_evolve_state(z_init,eig_f,t)
        print(np.abs(np.vdot(z_init,evolved_state))**2) 
        # if t<0.5:
            # return -100
        # else:
        return np.abs(np.vdot(z_init,evolved_state))**2

    def plot(state,t,H,use_syms=None):
        print("Plotting Fidelity...")
        if use_syms==None:
            z_ref = bin_to_int_base_m(state,H.system.base)
            z_index = find_index_bisection(z_ref,H.system.basis_refs)
            z_init = H.sector.eigvectors()[z_index,:]
            eig_f = H.sector.eigvalues()
        else:
            z_init,eig_f = energy_basis(state,H)

        fidelity_y = np.zeros(np.size(t))
        for n in range(0,np.size(t,axis=0)):
            evolved_state = time_evolve_state(z_init,eig_f,t[n])
            fidelity_y[n] = np.abs(np.vdot(evolved_state,z_init))**2

        plt.xlabel("t")
        plt.ylabel(r"$\vert \langle \psi(t) \vert \psi(0) \rangle \vert^2$")
        plt.title(str(H.system.base)+" Colour, N="+str(H.system.N))
        plt.plot(t,fidelity_y)
    def bulk_eval(state,t,H,use_syms=None):
        if use_syms==None:
            z_ref = bin_to_int_base_m(state,H.system.base)
            z_index = find_index_bisection(z_ref,H.system.basis_refs)
            z_init = H.sector.eigvectors()[z_index,:]
            eig_f = H.sector.eigvalues()
        else:
            z_init,eig_f = energy_basis(state,H)

        fidelity_y = np.zeros(np.size(t))
        for n in range(0,np.size(t,axis=0)):
            evolved_state = time_evolve_state(z_init,eig_f,t[n])
            fidelity_y[n] = np.abs(np.vdot(evolved_state,z_init))**2
        return fidelity_y

class entropy():
    def plot(H,k_vec=None):
        print("Plotting Entropy...")
        if k_vec is not None: 
            #get sym basis transformation, rotate eigstates from sym basis to prod basis
            #simpler to take bipartite entanglement in prod state basis
            U_mom = H.syms.basis_transformation(k_vec)
            eigvectors = H.sector.eigvectors(k_vec)
            eigvectors = np.dot(U_mom,eigvectors)
        else:
            eigvectors = H.sector.eigvectors(k_vec)
        basis_A_refs, basis_B_refs = bipartite_basis_split(H.system.basis,H.system.base,H.system.N)
        entropy = np.zeros(np.size(H.sector.eigvalues(k_vec)))
        pbar = ProgressBar()
        print("Calculating entropy of eigenstates")
        for n in pbar(range(0,np.size(H.sector.eigvalues(k_vec),axis=0))):
            state = H.sector.eigvectors(k_vec)[:,n]
            entropy[n] = entropy_half_chain_split(eigvectors[:,n],H.system.base,H.system.N,basis_A_refs,basis_B_refs)
        if k_vec is not None:
            plt.scatter(H.sector.eigvalues(k_vec),entropy,label=str(k_vec)+r" $\pi /$"+str(H.system.N)+" Symmetry sector",color="blue")
        else:
            plt.scatter(H.sector.eigvalues(k_vec),entropy)
        plt.xlabel("E")
        plt.ylabel("Entropy")
        plt.title("Eigenstate Bipartite Entropy, N="+str(H.system.N))
        plt.legend()
        # plt.show()
    def bulk_eval(self,H,k_vec=None):
        print("Plotting Entropy...")
        if k_vec is not None: 
            #get sym basis transformation, rotate eigstates from sym basis to prod basis
            #simpler to take bipartite entanglement in prod state basis
            U_mom = H.syms.basis_transformation(k_vec)
            eigvectors = H.sector.eigvectors(k_vec)
            eigvectors = np.dot(U_mom,eigvectors)
        else:
            eigvectors = H.sector.eigvectors(k_vec)
        basis_A_refs, basis_B_refs = bipartite_basis_split(H.system.basis,H.system.base,H.system.N)
        entropy = np.zeros(np.size(H.sector.eigvalues(k_vec)))
        pbar = ProgressBar()
        print("Calculating entropy of eigenstates")
        for n in pbar(range(0,np.size(H.sector.eigvalues(k_vec),axis=0))):
            state = H.sector.eigvectors(k_vec)[:,n]
            entropy[n] = entropy_half_chain_split(eigvectors[:,n],H.system.base,H.system.N,basis_A_refs,basis_B_refs)
        return entropy

def bipartite_basis_split(basis,base,N):
    N_A = int(np.floor(N/2))
    N_B = int(N-np.floor(N/2))
    #split basis
    basis_A = basis[:,:N_A]
    basis_B = basis[:,N_A:]
    basis_A_refs = np.zeros(np.size(basis,axis=0))
    basis_B_refs = np.zeros(np.size(basis,axis=0))
    for n in range(0,np.size(basis,axis=0)):
        basis_A_refs[n] = bin_to_int_base_m(basis_A[n],base)
        basis_B_refs[n] = bin_to_int_base_m(basis_B[n],base)
    return basis_A_refs,basis_B_refs

def entropy_half_chain_split(state,base,N,basis_A_refs,basis_B_refs):
    #generate coefficient matrix
    N_A = int(np.floor(N/2))
    N_B = int(N-np.floor(N/2))

    basis_A_refs_unique = np.unique(basis_A_refs)
    basis_B_refs_unique = np.unique(basis_B_refs)

    # M=np.zeros((np.power(base,N_A),np.power(base,N_B)),dtype=complex)
    M = np.zeros((np.size(basis_A_refs_unique),np.size(basis_B_refs_unique)),dtype=complex)

    #coefficients of state form matrix M_ij*state^A_i*state^V_j
    for n in range(0,np.size(state,axis=0)):
        if np.abs(state[n])>1e-4:
            i,j = int(basis_A_refs[n]),int(basis_B_refs[n])
            i_index,j_index = find_index_bisection(i,basis_A_refs_unique),find_index_bisection(j,basis_B_refs_unique)
            M[i_index,j_index] = M[i_index,j_index]+state[n]
    # print("M generated")
        
    #schmidt decomposition
    U,S,V = np.linalg.svd(M)
    to_del=[]
    for n in range(0,np.size(S,axis=0)):
        if S[n] == 0:
            to_del = np.append(to_del,n)
    for n in range(np.size(to_del,axis=0)-1,-1,-1):
        S=np.delete(S,to_del[n])
    # print("SVD Done")
        
    #von-neuman entropy
    entropy = -np.vdot(np.abs(S)**2,np.log(np.abs(S)**2))
    return entropy

def energy_basis(state,H):
    #find sym blocks state has non zero overlap in
    state_ref = bin_to_int_base_m(state,H.system.base)
    state_index = find_index_bisection(state_ref,H.system.basis_refs)
    sym_ref = H.syms.sym_data[state_index,0]
    k_refs = H.syms.find_k_ref(sym_ref)

    #dict to store state in sym sector basis
    z_sym=dict()
    eigenvalues=dict()
    for n in range(0,np.size(k_refs,axis=0)):
        psi = ref_state(state_ref,H.system)
        z_sym[n] = psi.sym_basis(k_refs[n],H.syms)
        eigenvalues[n] = H.sector.eigvalues(k_refs[n])

    #dict to store state from sym sector in energy eigenbasis
    z_energy=dict()
    for n in range(0,np.size(k_refs,axis=0)):
        z_energy[n] = np.zeros(np.size(H.sector.eigvalues(k_refs[n])),dtype = complex)
        #find coefficients by taking overlaps
        for m in range(0,np.size(z_energy[n],axis=0)):
            z_energy[n][m] = np.vdot(H.sector.eigvectors(k_refs[n])[:,m],z_sym[n])

    #combine all sym sectors
    z_init=z_energy[0]
    eig_f = eigenvalues[0]
    for n in range(1,len(z_energy)):
        z_init = np.append(z_init,z_energy[n])
        eig_f = np.append(eig_f,eigenvalues[n])
    return z_init,eig_f
