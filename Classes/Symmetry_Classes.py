#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from progressbar import ProgressBar
from State_Classes import sym_state

import sys
file_dir = '/home/kieran/Desktop/Work/CMP/physics_code/Exact_Diagonalization/functions'
sys.path.append(file_dir)
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m
from rw_functions import save_obj,load_obj

#generate set of nd-array elements to loop over
def nd_range(start, stop, dims):
  if not dims:
    yield ()
    return
  for outer in nd_range(start, stop, dims - 1):
    for inner in range(start, stop):
      yield outer + (inner,)

def is_int(number):
    if np.abs(number) < 1e-4:
        return True
    elif np.abs(number-np.round(number)) < 1e-4:
        return True
    else:
        return False

# class Symmetry:
    # def __init__(self,system,order=None):
        # self.system = system
        # self.order=order

class translational:
    def __init__(self,system):
        self.system = system
        self.sym_order = self.system.N
    def create_orbit(self,number):
        "generates all other states connected by translation"
        seq=[number]
        cycled_state = self.sym_op(number,1)
        while cycled_state != number:
            seq = np.append(seq,cycled_state)
            cycled_state=self.sym_op(cycled_state,1)
        return seq

    def sym_op(self,number,m):
        state = self.system.basis[self.system.keys[number]]
        L=np.size(state)
        for k in range(0,m):
            trans_state=np.zeros(np.size(state))
            trans_state[0:L-1] = state[1:L]
            trans_state[L-1] = state[0]
            # trans_state[1:L] = state[0:L-1]
            # trans_state[0] = state[L-1]
            state = trans_state
        return bin_to_int_base_m(state,self.system.base)

class translational_general:
    def __init__(self,system,order):
        self.system = system
        self.order = order
        self.sym_order = self.system.N/order
    def create_orbit(self,number):
        "generates all other states connected by translation"
        seq=[number]
        cycled_state = self.sym_op(number,1)
        while cycled_state != number:
            seq = np.append(seq,cycled_state)
            cycled_state=self.sym_op(cycled_state,1)
        return seq

    def sym_op(self,number,m):
        state = self.system.basis[self.system.keys[number]]
        L=np.size(state)
        for k in range(0,m):
            for n in range(0,self.order):
                trans_state=np.zeros(np.size(state))
                trans_state[0:L-1] = state[1:L]
                trans_state[L-1] = state[0]
                state = trans_state
        return bin_to_int_base_m(state,self.system.base)

class parity:
    def __init__(self,system):
        self.system = system
        self.sym_order = 2
    def create_orbit(self,state):
        state_bin = self.system.basis[self.system.keys[state]]
        state_pair = np.flip(state_bin,0)
        pair_ref = bin_to_int_base_m(state_pair,self.system.base)
        if pair_ref == state:
            return np.array((state))
        else:
            return np.array((state,pair_ref))

    def sym_op(self,number,m):
        if m % 2 == 0:
            return int(number)
        else:
            state = self.system.basis[self.system.keys[number]]
            parity_state = np.flip(state,0)
            parity_ref = bin_to_int_base_m(parity_state,self.system.base)
            return parity_ref

class PT:
    def __init__(self,system):
        self.system = system
        self.sym_order = 2
    def create_orbit(self,number):
        "generates all other states connected by translation"
        seq=[number]
        cycled_state = self.sym_op(number,1)
        while cycled_state != number:
            seq = np.append(seq,cycled_state)
            cycled_state=self.sym_op(cycled_state,1)
        # print("\n")
        # for n in range(0,np.size(seq,axis=0)):
            # print(self.system.basis[self.system.keys[seq[n]]])
        return seq

    def sym_op(self,number,m):
        state = self.system.basis[self.system.keys[number]]
        L=np.size(state)
        for k in range(0,m):
            trans_state=np.zeros(np.size(state))
            trans_state[0:L-1] = state[1:L]
            trans_state[L-1] = state[0]
            state = trans_state
            state = np.flip(state)
        return bin_to_int_base_m(state,self.system.base)

class inversion:
    def create_orbit(self,state):
        state_bin = self.system.basis[self.system.keys[state]]
        state_pair = np.ones(np.size(state_bin))-state_bin
        pair_ref = bin_to_int_base_m(state_pair,self.system.base)
        return pair_ref

    def sym_op(self,number,m):
        if m % 2 == 0:
            return int(number)
        else:
            state = self.system.basis[self.system.keys[number]]
            state_pair = np.ones(np.size(state))-state
            pair_ref = bin_to_int_base_m(state_pair,self.system.base)
            return pair_ref

class charge_conjugation:
    def create_orbit(self,state_ref):
        if state_ref == 0:
            return np.array((state_ref))
        else:
            conjugated_state = np.copy(self.system.basis[self.system.keys[state_ref]])
            for m in range(0,np.size(conjugated_state,axis=0)):
                if conjugated_state[m] != 0:
                    conjugated_state[m] = (self.system.base-conjugated_state[m]) % self.system.base
            conjugated_ref = bin_to_int_base_m(conjugated_state,self.system.base)
            return np.array((state_ref,conjugated_ref))
    def sym_op(self,state_ref,m):
        if m % 2 == 0:
            return int(state_ref)
        else:
            conjugated_state = np.copy(self.system.basis[self.system.keys[state_ref]])
            for m in range(0,np.size(conjugated_state,axis=0)):
                if conjugated_state[m] != 0:
                    conjugated_state[m] = (self.system.base-conjugated_state[m]) % self.system.base
            conjugated_ref = bin_to_int_base_m(conjugated_state,self.system.base)
            return conjugated_ref

class model_sym_data:
    def __init__(self,system,syms):
        self.system=system
        self.syms = syms

        #multi-dim array dimensions from order of each symmetry
        order_ranges = dict()
        for k in range(0,np.size(self.syms,axis=0)):
            order_ranges[k] = np.arange(0,self.syms[k].sym_order).astype(int)
        order_ranges_list = []
        for k in range(0,len(order_ranges)):
            order_ranges_list.append(order_ranges[k])
        import itertools
        self.orbit_indices = list(itertools.product(*order_ranges_list))
        # self.orbit_indices = list(nd_range(0,self.system.N,np.size(self.syms)))
        self.sym_data,self.orbit_array_store,self.sym_data_per_ref,self.sym_data_ref_periods = self.find_joint_sym_data()
        self.sym_refs = np.sort(np.unique(self.sym_data[:,0]))

    def find_joint_sym_data(self):
        "for each number in basis (0,2^N-1), finds lowest trans equiv ref state and jumps needed to get there"
        basis_refs = self.system.basis_refs
        N = self.system.N

        array_dims = (self.system.N) * np.ones(np.size(self.syms),dtype=int)
        low_refs = np.zeros((np.size(basis_refs,axis=0),np.size(self.syms)+2))
        found=[]
        pbar = ProgressBar(maxval=self.system.dim)
        pbar.start()
        print("Finding unique reference states")

        orbit_array_store=dict()
        sym_data_per_ref=dict()
        sym_data_ref_periods=dict()
        for n in range(0,np.size(basis_refs,axis=0),1):
            #if not in found, generate multi-dimensional array containing all states
            #connected by symmetry operations, A[n1,n2,...] = T_1^n1 * ... |a>
            if basis_refs[n] not in found:
                periods=[]
                for m in range(0,np.size(self.syms,axis=0)):
                    periodicity = np.size(self.syms[m].create_orbit(basis_refs[n]))
                    periods = np.append(periods,periodicity)
                allowed_period_k = 2*math.pi/periods
                sym_data_ref_periods[basis_refs[n]] = periods

                #vector to save list of state ref that are in the same orbit, and the L vector connecting the reference state
                sym_data_per_ref[basis_refs[n]] = np.zeros((1,1+np.size(self.syms,axis=0))) #init row
                # sym_data_per_ref[basis_refs[n]][0,0] = basis_refs[n]

                orbit_array = basis_refs[n]*np.ones(array_dims)
                orbit_phase_array = np.zeros(array_dims)
                #state to be updated with +1 coefficient of orbit state (ie 0 mom k state)
                orbit_state = np.zeros(np.size(basis_refs),dtype=complex)
                # orbit_state[n] = 1
                sym_ref = basis_refs[n]
                for m in range(0,np.size(self.orbit_indices,axis=0)):
                    ref = basis_refs[n]
                    for k in range(0,np.size(self.syms,axis=0)):
                        ref = self.syms[k].sym_op(ref,self.orbit_indices[m][k])

                    # only consider first orbit:
                    in_first_orbit = 1
                    for count in range(0,np.size(self.orbit_indices[m],axis=0)):
                        if self.orbit_indices[m][count] >= periods[count]:
                            in_first_orbit = 0
                            break
                    if in_first_orbit == 1:
                        temp = np.append(ref,self.orbit_indices[m])
                        sym_data_per_ref[basis_refs[n]] = np.vstack((sym_data_per_ref[basis_refs[n]],temp))

                    ref_index = self.system.keys[ref]

                    # if ref != basis_refs[n]:
                    orbit_state[ref_index] = orbit_state[ref_index] + 1 #form norm, used in H
                    orbit_array[self.orbit_indices[m]] = ref

                sym_data_per_ref[basis_refs[n]] = np.delete(sym_data_per_ref[basis_refs[n]],0,axis=0)

                #data to generate Hamiltonian
                Norm = np.real(np.power(np.vdot(orbit_state,orbit_state),0.5))
                orbit_ref = np.min(orbit_array)
                orbit_array_store[orbit_ref] = orbit_array
                L = np.array(np.unravel_index(orbit_array.argmin(), orbit_array.shape))
                #loop through and add data of all states in orbit, update found (skip)
                # for m in range(0,np.size(self.orbit_indices,axis=0)):
                for m in range(0,np.size(sym_data_per_ref[basis_refs[n]],axis=0)):
                    ref = sym_data_per_ref[basis_refs[n]][m,0]
                    if ref not in found:
                        found = np.unique(np.sort(np.append(found,ref)))
                        pbar.update(np.size(found))
                        L_ref = sym_data_per_ref[basis_refs[n]][m,1:]

                        index = self.system.keys[ref]
                        low_refs[index,0] = orbit_ref
                        low_refs[index,1] = Norm
                        low_refs[index,2:] = L_ref
        pbar.finish()
        return low_refs, orbit_array_store,sym_data_per_ref,sym_data_ref_periods

    def sym_state(self,ref,k_vec):
        state_sym_data = self.sym_data_per_ref[ref]
        state = np.zeros(np.size(self.system.basis_refs),dtype=complex)
        #check fits in k_vec first
        fits_in_orbit = 1
        periodicities = self.sym_data_ref_periods[ref]
        for m in range(0,np.size(periodicities,axis=0)):
            if is_int(float(periodicities[m]*k_vec[m])/self.system.N) == False: #if state fits
                fits_in_orbit = 0
                break
        if fits_in_orbit ==0:
            return state
        else:
            for n in range(0,np.size(state_sym_data,axis=0)):
                index = self.system.keys[state_sym_data[n,0]] #get key from ref
                state[index] = state[index] + np.exp(1j*2*math.pi/self.system.N*np.dot(state_sym_data[n,1:],k_vec))
            return state


    #find all ref state (lowest orbit ref) in a symmetry block
    def find_block_refs(self,k_vec):
        block_refs  = []
        for n in range(0,np.size(self.sym_refs,axis=0)):
            sym_state_prod_basis = self.sym_state(self.sym_refs[n],k_vec)
            if (np.abs(sym_state_prod_basis)<1e-5).all() == False:  #if state not zero
                block_refs = np.append(block_refs,self.sym_refs[n])
        return block_refs

    #find all sym blocks (quantum numbers k_vec) that a (lowest orbit ref) state is in
    def find_k_ref(self,ref):
        k_refs = np.zeros(np.size(self.syms))
        sym_ref = self.sym_data[self.system.keys[ref],0]

        #all poss sym Q numbers
        quant_numbers = list(nd_range(0,self.system.N,np.size(self.syms)))
        pbar=ProgressBar()
        for m in range(0,np.size(quant_numbers,axis=0)):
            sym_state_prod_basis = self.sym_state(sym_ref,quant_numbers[m])
            if (np.abs(sym_state_prod_basis)<1e-5).all() == False:  #if state not zero
                k_refs = np.vstack((k_refs,quant_numbers[m]))
        #delete init row
        k_refs = np.delete(k_refs,0,axis=0)
        return k_refs

    #full basis transformation U
    def basis_transformation(self,k_vec):
        temp = self.sym_state(self.sym_refs[0],k_vec)
        U = np.zeros(np.size(temp)) 
        pbar = ProgressBar()
        print("Finding transformation of "+str(k_vec)+" sector states to product states")
        for n in pbar(range(0,np.size(self.sym_refs,axis=0))):
            state = self.sym_state(self.sym_refs[n],k_vec)
            if (np.abs(state)<1e-5).all() == False:  #if state not zero
                state = state / np.power(np.vdot(state,state),0.5)
                U = np.vstack((U,state))
        U = np.delete(U,0,axis=0)
        return np.transpose(U)
