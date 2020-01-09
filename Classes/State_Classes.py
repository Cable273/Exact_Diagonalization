#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from progressbar import ProgressBar
from System_Classes import unlocking_System
import itertools

import sys
file_dir = '/home/kieran/Desktop/Work/CMP/physics_code/Exact_Diagonalization/functions'
sys.path.append(file_dir)
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection

def nd_range(start, stop, dims):
  if not dims:
    yield ()
    return
  for outer in nd_range(start, stop, dims - 1):
    for inner in range(start, stop):
      yield outer + (inner,)

class prod_state:
    def prod_basis(self):
        v = np.zeros(np.size(self.system.basis_refs))
        index = find_index_bisection(self.ref,self.system.basis_refs)
        v[index] = 1
        return v

    def energy_basis(self,H,k_refs=None):
        if k_refs is None:
            self.energy_rep_eigs = H.sector.eigvalues()
            self.energy_rep = np.conj(H.sector.eigvectors()[self.key,:])
            return self.energy_rep
        else:
            #dict to store state in sym sector basis
            k_refs = H.syms.find_k_ref(self.ref)
            z_sym=dict()
            eigenvalues=dict()
            for n in range(0,np.size(k_refs,axis=0)):
                z_sym[n] = self.sym_basis(k_refs[n],H.syms)
                eigenvalues[n] = H.sector.eigvalues(k_refs[n])

            #dict to store state from sym sector in energy eigenbasis
            z_energy=dict()
            for n in range(0,np.size(k_refs,axis=0)):
                z_energy[n] = np.zeros(np.size(H.sector.eigvalues(k_refs[n])),dtype = complex)
                #find coefficients by taking overlaps
                for m in range(0,np.size(z_energy[n],axis=0)):
                    z_energy[n][m] = np.vdot(H.sector.eigvectors(k_refs[n])[:,m],z_sym[n])

            #combine all sym sectors
            z_energy_comb=z_energy[0]
            eig_comb = eigenvalues[0]
            for n in range(1,len(z_energy)):
                z_energy_comb = np.append(z_energy_comb,z_energy[n])
                eig_comb = np.append(eig_comb,eigenvalues[n])
            self.energy_rep_eigs = eig_comb
            self.energy_rep = z_energy_comb
            return z_energy_comb

    #|a> = C_{k1,k2...}|a_ref,k1,k2,...>
    # returns (0,0,...,C_{k1,k2...},..,0) array for symetry block
    def sym_basis(self,k_vec,sym_data):
        #permutations of all array indices for d dimensional arrayk, d=no syms
        #for looping through d-dim array
        orbit_indices = list(nd_range(0,self.system.N,np.size(sym_data.syms)))

        #lowest symmetry equiv state, the reference
        # index = self.system.keys[self.ref]
        sym_ref = sym_data.sym_data[self.system.keys[self.ref],0]
        L = sym_data.sym_data[self.system.keys[self.ref],2:]
        allowed_mom = np.zeros(np.size(k_vec))

        periods = np.zeros(np.size(sym_data.syms))
        for k in range(0,np.size(sym_data.syms,axis=0)):
            temp_orbit = sym_data.syms[k].create_orbit(sym_ref)
            periods[k] = np.size(temp_orbit)

        psi = sym_state(self.ref,k_vec,sym_data,self.system)
        temp = psi.prod_basis()

        orbit_size = 0
        orbit_ref_indices = []
        for n in range(0,np.size(temp,axis=0)):
            if np.abs(temp[n]) > 1e-5:
                orbit_size = orbit_size + 1
                orbit_ref_indices = np.append(orbit_ref_indices,n)

        U = np.zeros(orbit_size,dtype=complex)

        k_refs = sym_data.find_k_ref(sym_ref)
            
        #loop through all sym sectors and keep |a_ref,k1,...kn>=c_n|n> to form U
        c,d  = 0,0
        for m in range(0,np.size(k_refs,axis=0)):
            psi = sym_state(self.ref,k_refs[m],sym_data,self.system)
            temp = psi.prod_basis()
            U_temp = np.zeros(orbit_size,dtype=complex)
            for k in range(0,np.size(orbit_ref_indices,axis=0)):
                U_temp[k] = temp[int(orbit_ref_indices[k])]
            U = np.vstack((U,U_temp))

        not_found = 1
        for n in range(0,np.size(k_refs,axis=0)):
            if (k_refs[n] == k_vec).all() == True:
                j_index = n
                not_found = 0
                break
        if not_found == 1:
            print("Error: State not in this symmetry sector")
        else:
            U = np.delete(U,0,axis=0)

            #find column index of relevant state
            for n in range(0,np.size(orbit_ref_indices,axis=0)):
                if orbit_ref_indices[n] == self.key:
                    i_index = n

            #delete equivalent rows
            eq_ind=np.arange(0,np.size(k_refs,axis=0)) #track what quant sectors are equiv
            for n in range(0,np.size(U,axis=0)):
                to_del = []
                for m in range(n+1,np.size(U,axis=0)):
                    if (np.abs(U[m]-U[n])<1e-5).all() == True:
                        to_del = np.append(to_del,m)
                        eq_ind[m] = n
                    elif (np.abs(1j * U[m]-U[n])<1e-5).all() == True:
                        to_del = np.append(to_del,m)
                        eq_ind[m] = n
                    elif (np.abs(-1j * U[m]-U[n])<1e-5).all() == True:
                        to_del = np.append(to_del,m)
                        eq_ind[m] = n
                    elif (np.abs(-1 * U[m]-U[n])<1e-5).all() == True:
                        to_del = np.append(to_del,m)
                        eq_ind[m] = n
                for m in range(np.size(to_del,axis=0)-1,-1,-1):
                    U = np.delete(U,to_del[m],axis=0)

            #update j_index, deleted eq rows
            j_index = eq_ind[j_index]
            found=[]
            keys=dict()
            c = 0
            for n in range(0,np.size(eq_ind,axis=0)):
                if eq_ind[n] not in found:
                    keys[eq_ind[n]]=c
                    c = c+1
                    found = np.append(found,eq_ind[n])
            j_index = keys[j_index]

            #inv transformation
            U_inv = np.linalg.inv(U)
            coef = U_inv[i_index,j_index]

            block_references = sym_data.find_block_refs(k_vec)
            prod_state = np.zeros(np.size(block_references),dtype=complex)
            # ref_index = self.system.keys[sym_ref]
            ref_index = find_index_bisection(sym_ref,block_references)
            prod_state[ref_index] = coef
            return prod_state

class zm_state(prod_state):
    def __init__(self,order,pol,system,shift=0):
        self.system = system
        self.order = order

        #sets the bit representation |1,0...>
        zm=np.zeros(self.system.N)
        zm[0] = pol
        count=1
        for n in range(1,np.size(zm,axis=0)):
            if count<order:
                zm[n] = 0
                count = count + 1
            else:
                zm[n] = pol
                count = 1
        #count zeros at end to check state is act z_n
        no_zeros=0
        for n in range(np.size(zm,axis=0)-1,-1,-1):
            if zm[n] == pol:
                break
            no_zeros = no_zeros + 1
        if no_zeros == order-1:
            self.bits = zm
        else: 
            print("ERROR Z_"+str(order)+" not at this chain length.")

        for n in range(0,shift):
            self.bits = cycle_bits_state(self.bits)
        self.ref = bin_to_int_base_m(self.bits,self.system.base)
        self.key = system.keys[self.ref]

class bin_state(prod_state):
    def __init__(self,bits,system):
        self.bits = bits
        self.system = system
        self.ref = bin_to_int_base_m(self.bits,self.system.base)
        self.key = system.keys[self.ref]

class ref_state(prod_state):
    def __init__(self,ref,system):
        self.system = system
        self.ref = ref
        self.bits = int_to_bin_base_m(self.ref,self.system.base,self.system.N)
        self.key = system.keys[self.ref]

#|a,k1,k2,...,km>=
class sym_state:
    def __init__(self,ref,k_vec,model_sym_data,system):
        self.model_sym_data = model_sym_data
        self.ref = ref  
        self.k_vec = k_vec
        self.system = system

    #|a,k1,k2..> = a_n |n>
    def sym_basis(self):
        #get smallest ref in orbit (represenatative state)
        index = self.system.keys[self.ref]
        sym_ref = self.model_sym_data.sym_data[index,0]

        block_ref_states = self.model_sym_data.find_block_refs(self.k_vec)
        block_ref_keys=dict()
        for n in range(0,np.size(block_ref_states,axis=0)):
            block_ref_keys[block_ref_states[n]] = n
            
        if sym_ref in block_ref_states:
            ref_index = block_ref_keys[sym_ref]
            v = np.zeros(np.size(block_ref_states))
            v[ref_index] = 1
            return v
        else:
            print("Error: State not in this symmetry sector")

    #sym_state in prod state basis ie |ref,k1,k2,..> = a_n |n>
    def prod_basis(self):
        #generate set of nd-array elements to loop over
        array_dims = (self.system.N) * np.ones(np.size(self.model_sym_data.syms),dtype=int)
        orbit_array = np.zeros(array_dims)
        phase_array = np.zeros(array_dims,dtype=complex)
        orbit_indices = list(nd_range(0,self.system.N,np.size(self.model_sym_data.syms)))
        #state to be updated with +1 coefficient of orbit state (ie 0 mom k state)
        orbit_state = np.zeros(np.size(self.system.basis_refs),dtype=complex)

        sym_ref = self.model_sym_data.sym_data[self.system.keys[self.ref],0]
        block_ref_states = self.model_sym_data.find_block_refs(self.k_vec)
        if sym_ref in block_ref_states:
            for m in range(0,np.size(orbit_indices,axis=0)):
                ref = sym_ref
                for k in range(0,np.size(self.model_sym_data.syms,axis=0)):
                    ref = self.model_sym_data.syms[k].sym_op(ref,orbit_indices[m][k])
                orbit_array[orbit_indices[m]] = ref
                phase_array[orbit_indices[m]] = np.exp(1j*2*math.pi/self.system.N * np.dot(self.k_vec,orbit_indices[m]))
                ref_index = self.system.keys[ref]
                orbit_state[ref_index] = orbit_state[ref_index] + phase_array[orbit_indices[m]]

        if (orbit_state==np.zeros(np.size(orbit_state))).all() == False:
            orbit_state = orbit_state * np.power(np.vdot(orbit_state,orbit_state),-0.5)
        return orbit_state


def full_state_map(subsystem_bit_rep,site_labels,system,subsystem):
    full_prod_state_bits = np.zeros(system.N)
    for n in range(0,np.size(site_labels,axis=0)):
        full_prod_state_bits[site_labels[n]] = subsystem_bit_rep[n]
    full_ref = bin_to_int_base_m(full_prod_state_bits,system.base)
    return full_ref

#Form a state made from the polarized state |000...> with n singlets added.
#Form equal superposition over all possible ways of adding n singlets.
#Final state will be a |J,-J> state, with J=L/2 - n
def total_spin_state(J,system):
    s_system = 0.5*(system.base-1)
    J_max = system.N*s_system
    no_singlets = int(J_max-J)
    if no_singlets == 0:
        psi = np.zeros(np.size(system.basis_refs))
        psi[0] = 1
        return psi
    else:
        #form simplest subsystem wf of n singlets next to each other
        subsystem = unlocking_System([0,1],"periodic",2,2*no_singlets)
        if no_singlets>1:
            psi_sub0 = np.zeros(np.size(subsystem.basis_refs))
            for n in range(0,np.size(subsystem.basis_refs,axis=0)):
                state = np.copy(subsystem.basis[n])
                phase=1
                for m in range(0,np.size(state,axis=0),2):
                    temp = np.array((state[m],state[m+1]))
                    if (temp == np.array((0,0))).all() or (temp == np.array((1,1))).all():
                        phase = 0
                        break
                    elif (temp == np.array((0,1))).all():
                        phase = phase * -1
                psi_sub0[n] = phase
        else:
            psi_sub0 = np.zeros(np.size(subsystem.basis_refs))
            psi_sub0[2] = 1
            psi_sub0[1] = -1
        psi_sub0 = psi_sub0/np.power(np.vdot(psi_sub0,psi_sub0),0.5)

        #get all locations (relabelling) singlets can be located
        sites = np.arange(0,system.N)
        singlet_sites = np.array((list(itertools.combinations(sites,2*no_singlets))))
        from combinatorics import all_pairs
        poss_singlet_loc=np.zeros((np.size(singlet_sites,axis=0),no_singlets*2),dtype=int)

        for n in range(0,np.size(singlet_sites,axis=0)):
            for pairs in all_pairs(list(singlet_sites[n])):
                set_of_pairs = np.ndarray.flatten(np.array((pairs)))
                poss_singlet_loc[n] = set_of_pairs


        # #form binary prod states from psi_sub0 and permuting.
        # #|J,-J> formed by equal superposition of all states, remaining sites 0
        psi = np.zeros(np.size(system.basis_refs))
        print("Forming |J,-J> state, J=",J)
        pbar=ProgressBar()
        for n in pbar(range(0,np.size(poss_singlet_loc,axis=0))):
            for m in range(0,np.size(psi_sub0,axis=0)):
                ref = full_state_map(subsystem.basis[m],poss_singlet_loc[n],system,subsystem)
                psi[system.keys[ref]] = psi[system.keys[ref]] + psi_sub0[m]

        psi = psi/np.power(np.vdot(psi,psi),0.5)
        return psi

#choose
import operator as op
from functools import reduce
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom
#create coherent states to init trajectories from
def coherent_state(alpha,su2_states,J):
    wf_su2 = np.zeros(np.size(su2_states,axis=1),dtype=complex)

    eta = alpha*np.tan(np.abs(alpha))/np.abs(alpha)
    for m in range(0,np.size(wf_su2,axis=0)):
        wf_su2[m] = np.power(ncr(int(2*J),m),0.5)*np.power(eta,m)

    wf = wf_su2[0] * su2_states[:,0]
    for m in range(1,np.size(wf_su2,axis=0)):
        wf = wf + wf_su2[m] * su2_states[:,m]

    wf = wf / np.power(1+np.abs(eta)**2,J)
    return wf

