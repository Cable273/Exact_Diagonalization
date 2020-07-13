#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m
from Search_functions import find_index_bisection
import pandas as pd
from progressbar import ProgressBar
from copy import deepcopy

import sys
file_dir = '/home/kieran/Desktop/Work/CMP/physics_code/Exact_Diagonalization/functions'
sys.path.append(file_dir)
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m
from Search_functions import find_index_bisection

class U1_system:
    def __init__(self,bc,base,N,m):
        self.bc = bc 
        self.base = base
        self.N = N
        self.m = m

    def gen_basis(self,M):
        from itertools import product,permutations
        dim = np.power(self.N,self.base)
        poss_occs = np.array(list(product(np.arange(0,self.N),repeat=self.base)))
        #delete occs not satisfying sum_i n_i = N
        to_del=[]
        for n in range(0,np.size(poss_occs,axis=0)):
            if np.sum(poss_occs[n]) != self.N:
                to_del = np.append(to_del,n)
        for n in range(np.size(to_del,axis=0)-1,-1,-1):
            poss_occs=np.delete(poss_occs,to_del[n],axis=0)

        #delete occs not satisfying sum_i n_i m_i = M
        to_del=[]
        for n in range(0,np.size(poss_occs,axis=0)):
            if np.abs(np.dot(self.m,poss_occs[n])) != M:
                to_del = np.append(to_del,n)
        for n in range(np.size(to_del,axis=0)-1,-1,-1):
            poss_occs=np.delete(poss_occs,to_del[n],axis=0)

        def unique_perm(series):
            return {"".join(p) for p in permutations(series)}
            
        #now find all ways to distribute bits to generate product basis in U(1) sector
        self.basis = np.zeros(self.N)
        pbar=ProgressBar()
        print("Generating U(1) sector basis")
        for n in pbar(range(0,np.size(poss_occs,axis=0))):
            string = []
            for m in range(0,np.size(poss_occs[n],axis=0)):
                string = np.append(string,m*np.ones(poss_occs[n,m]))
            bit_dist = np.array(list(set(permutations(string))))
            self.basis = np.vstack((self.basis,bit_dist))
        self.basis = np.delete(self.basis,0,axis=0)

        #update basis info
        self.basis_refs = np.zeros(np.size(self.basis,axis=0))
        self.keys=dict()
        for n in range(0,np.size(self.basis_refs,axis=0)):
            self.basis_refs[n] = bin_to_int_base_m(self.basis[n],self.base)
            self.keys[self.basis_refs[n]] = n
        self.dim = np.size(self.basis_refs)


class unlocking_System:
    def __init__(self,unlockers,bc,base,N):
        self.bc = bc
        self.base = base
        self.N = N
        self.unlockers = unlockers

    def gen_basis(self):
        self.basis = self.gen_P0K_basis()
        self.basis_refs = np.zeros(np.size(self.basis,axis=0))
        self.keys = dict()
        for n in range(0,np.size(self.basis,axis=0)):
            self.basis_refs[n] = bin_to_int_base_m(self.basis[n],self.base)
            self.keys[self.basis_refs[n]] = n
        self.dim = np.size(self.basis_refs)

    def gen_P0K_basis(self):
        #to append 0,...,K-1 to current basis and grow it recursively
        def blocking_append(states,unlockers,base,check_pbc):
            #each G,B surrounded by R
            new_states = dict()
            for n in range(0,base):
                new_states[n] = np.zeros((np.size(states,axis=0),np.size(states,axis=1)+1),dtype=int)

            for n in range(0,np.size(states,axis=0)):
                new_states[0][n,1:] = states[n] 
                if states[n,0] in unlockers:
                    if check_pbc == 1:
                        if states[n,np.size(states,axis=1)-1] in unlockers:
                            for m in range(1,base):
                                new_states[m][n,1:] = states[n]
                    else:
                        for m in range(1,base):
                            new_states[m][n,1:] = states[n]

            for n in range(0,base):
                new_states[n][:,0] = n
            new_states_full = new_states[0]
            for n in range(1,base):
                new_states_full = np.unique(np.vstack((new_states_full,new_states[n])),axis=0)
            return new_states_full

        #generate basis
        print("Generating Constrained basis")
        v=np.arange(0,self.base).reshape(self.base,1)
        if self.N==2:
            return blocking_append(v,self.unlockers,self.base,0)
        else:
            for n in range(0,self.N-1):
                if n != self.N-2:
                    v = blocking_append(v,self.unlockers,self.base,0)
                else:
                    if self.bc == "periodic":
                        v = blocking_append(v,self.unlockers,self.base,1)
                    elif self.bc == "open":
                        v = blocking_append(v,self.unlockers,self.base,0)
            print("Basis Generated, dim="+str(np.size(v,axis=0)))
        return v

    def U1_sector(self,n_up):
        from sympy.utilities.iterables import multiset_permutations
        root = np.append(np.ones(n_up),np.zeros(self.N-n_up)).astype(int)
        basis = np.array(list(multiset_permutations(root)))

        new_basis = deepcopy(self)
        new_basis.basis = basis
        new_basis.basis_refs = np.zeros(np.size(new_basis.basis,axis=0))
        new_basis.keys = dict()
        for n in range(0,np.size(new_basis.basis_refs)):
            new_basis.basis_refs[n] = bin_to_int_base_m(new_basis.basis[n],new_basis.base)
            new_basis.keys[new_basis.basis_refs[n]] = n
        new_basis.dim = np.size(new_basis.basis_refs)
        return new_basis

class f1_System:
    def __init__(self,name,bc,N):
        self.base = 3 #3level system
        self.name = name
        self.bc = bc
        self.N = N

        self.basis = self.gen_basis()
			
        self.basis_refs = np.zeros(np.size(self.basis,axis=0),dtype=int)
        self.keys = dict()
        for n in range(0,np.size(self.basis,axis=0)):
            self.basis_refs[n] = bin_to_int_base_m(self.basis[n],self.base)
            self.keys[self.basis_refs[n]] = n

    def gen_basis(self):
        #to append 0,...,K-1 to current basis and grow it recursively
        basis = np.arange(0,self.base).reshape(self.base,1)

        if self.bc == "open":
            N_max = self.N+1
        else: #do end explicitly for periodic
            N_max = self.N

        #initial growth:
        for site in range(2,N_max):
            grown_basis = np.zeros(site)
            for n in range(0,np.size(basis,axis=0)):
                blocked=[]
                R_neighbour = basis[n,0]
                if R_neighbour == 0:
                    blocked = np.append(blocked,1)
                elif R_neighbour == 1:
                    blocked = np.append(blocked,0)

                for m in range(0,self.base):
                    if m not in blocked:
                        grown_basis = np.vstack((grown_basis,np.append(np.array((m)),basis[n])))
            basis = np.delete(grown_basis,0,axis=0)

        if self.bc =="periodic":
            grown_basis = np.zeros(self.N)
            for n in range(0,np.size(basis,axis=0)):
                L_neighbour = int(basis[n,0])
                R_neighbour = int(basis[n,np.size(basis,axis=1)-1])
                blocked = []
                if L_neighbour == 0  or R_neighbour == 0:
	                blocked = np.append(blocked,1)
                if L_neighbour == 1  or R_neighbour == 1:
	                blocked = np.append(blocked,0)
                blocked = np.unique(blocked)

                for m in range(0,self.base):
	                if m not in blocked:
		                grown_basis = np.vstack((grown_basis,np.append(np.array((m)),basis[n])))
            basis = np.delete(grown_basis,0,axis=0)

        return basis

class unlocking_System_DoubleBlock:
    def __init__(self,unlocker,bc,base,N):
        self.bc = bc
        self.base = base
        self.N = N
        self.unlocker = unlocker

    def gen_basis(self):
        self.basis = self.gen_doubeRydbergBasis()
        self.basis_refs = np.zeros(np.size(self.basis,axis=0))
        self.keys = dict()
        for n in range(0,np.size(self.basis,axis=0)):
            self.basis_refs[n] = bin_to_int_base_m(self.basis[n],self.base)
            self.keys[self.basis_refs[n]] = n
        self.dim = np.size(self.basis_refs)

    def gen_doubeRydbergBasis(self):
        #keep appending to dictionary, growing recursively until chain size reached
        #length 1+2
        states_length_n = dict()
        states_length_n[1] = dict()
        for n in range(0,self.base):
            states_length_n[1][n] = np.array([n])

        states_length_n[2] = dict()
        c=0
        for n in range(0,len(states_length_n[1])):
            if states_length_n[1][n][0] == self.unlocker:
                for m in range(0,self.base):
                    new_state = np.append([m],states_length_n[1][n])
                    if np.size(new_state)<=self.N:
                        states_length_n[2][c] = new_state
                        c+=1
            else:
                new_state = np.append([self.unlocker,self.unlocker],states_length_n[1][n])
                if np.size(new_state)<=self.N:
                    states_length_n[2][c] = new_state
                    c+=1

        for length in range(3,self.N+1):
            states_length_n[length] = dict()
            c=0
            for n in range(0,len(states_length_n[length-1])):
                if np.size(states_length_n[length-1][n]) < self.N:
                    #append any state
                    if states_length_n[length-1][n][0] == self.unlocker and states_length_n[length-1][n][1] == self.unlocker:
                        for m in range(0,self.base):
                            new_state = np.append([m],states_length_n[length-1][n])
                            if np.size(new_state)<=self.N:
                                states_length_n[length][c] = new_state
                                c+=1
                    #append only double + sing unlocker
                    else:
                        new_state = np.append([self.unlocker,self.unlocker],states_length_n[length-1][n])
                        if np.size(new_state)<=self.N:
                            states_length_n[length][c] = new_state
                            c+=1
                        new_state = np.append([self.unlocker],states_length_n[length-1][n])
                        if np.size(new_state)<=self.N:
                            states_length_n[length][c] = new_state
                            c+=1
                else:
                    states_length_n[length][c] = states_length_n[length-1][n]
                    c+=1
            del states_length_n[length-2]


        print("basis grown, trimming")
        basis = np.zeros((len(states_length_n[self.N]),self.N))
        c=0
        pbar=ProgressBar()
        for n in pbar(range(0,len(states_length_n[self.N]))):
            if self.bc == "periodic":
                if states_length_n[self.N][n][0] == self.unlocker and states_length_n[self.N][n][1] == self.unlocker:
                    basis[c] = states_length_n[self.N][n]
                    c+=1
                elif states_length_n[self.N][n][np.size(states_length_n[self.N][n])-1] == self.unlocker and states_length_n[self.N][n][np.size(states_length_n[self.N][n])-2] == self.unlocker:
                    basis[c] = states_length_n[self.N][n]
                    c+=1
                elif states_length_n[self.N][n][0] == self.unlocker and states_length_n[self.N][n][np.size(states_length_n[self.N][n])-1] == self.unlocker:
                    basis[c] = states_length_n[self.N][n]
                    c+=1
            else:
                basis[c] = states_length_n[self.N][n]
                c+=1
        basis = np.unique(basis,axis=0)
        return basis
