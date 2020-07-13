#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from progressbar import ProgressBar
from copy import deepcopy

import sys
file_dir = '/home/kieran/Desktop/Work/CMP/physics_code/Exact_Diagonalization/functions'
sys.path.append(file_dir)
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m
from Search_functions import find_index_bisection

def prod(v1,v2,base):
    #find tensor product of v1,v2. For use with eg gen_clock_H
    #must be used with full base^N basis,can project after
    dim1=np.size(v1)
    dim2=np.size(v2)
    N1=int(np.log2(dim1))
    N2=int(np.log2(dim2))
    dim = np.power(base,N1+N2)

    M=np.outer(v1,v2)
    out=np.zeros(dim,dtype=complex)
    for n in range(0,np.size(M,axis=0)):
        for m in range(0,np.size(M,axis=1)):
            state1=int_to_bin_base_m(n,base,N1)
            state2=int_to_bin_base_m(m,base,N2)
            state = np.append(state1,state2)
            ref = int(bin_to_int_base_m(state,base))
            out[ref] = out[ref] + M[n,m]
    return out

class H_operations:
    def add(H1,H2,coef):
        H = Hamiltonian(H1.system,H1.syms)
        keys = H1.sector.table.keys()
        for key in keys:
            H.sector.table[key] = H_entry(coef[0]*H1.sector.table[key].H+coef[1]*H2.sector.table[key].H)
        return H

#Class to generate Hamiltonians. H_table for storing them
class Hamiltonian:
    def __init__(self,system,syms=None):
        self.system = system
        self.sector = H_table(self.system)
        self.syms=syms
        self.site_ops=dict()
        self.dim = dict()

        self.uc_size = False
        self.uc_pos = False

    def gen(self,k_vec=None,staggered=False):
        #construct H by looping through basis and acting on state with each H_i ,H = sum_i (H_i)
        op_sizes=np.zeros(np.size(self.model,axis=0))
        for n in range(0,np.size(op_sizes,axis=0)):
            op_sizes[n] = np.size(self.model[n])
            
        if k_vec is None: #no sym use full basis
            block_references = self.system.basis_refs
            block_keys = self.system.keys
        else: #use symmetry block basis
            print("Sector: "+str(k_vec))
            block_references = self.syms.find_block_refs(k_vec)
            block_keys = dict()
            for m in range(0,np.size(block_references,axis=0)):
                block_keys[block_references[m]] = m
                
        dim = np.size(block_references)
        print("Dim="+str(dim))
        pbar = ProgressBar()
        #loop through basis states
        for i in pbar(range(0,dim)):
        # for i in range(0,dim):
            #state = bit representation of product state
            state = self.system.basis[self.system.keys[block_references[i]]]
            self.update_H_pos_sweep(state,i,block_references,block_keys,op_sizes,k_vec=k_vec)

    def update_H_pos_sweep(self,state,i_index,block_references,block_keys,op_sizes,k_vec=None):
        new_refs_coef = dict() #for storing all refs and there coef from looping positions
        for position in range(0,self.system.N):
            #indices operators act on (ie check pbc and loop around if at edge)

            #periodic site_indices (wrap around chain)
            if self.system.bc == "periodic":
                site_indices = dict()
                for op_index in range(0,np.size(self.model,axis=0)):
                    d=np.size(self.model[op_index])
                    site_indices[op_index] = np.zeros(d)
                    for n in range(0,d):
                        if position+n < self.system.N:
                            site_indices[op_index][n] = position+n
                        else:
                            site_indices[op_index][n] = position+n-self.system.N
            #open (convention: Just throw away all terms that would wrap around chain)
            #eg, nn int x_i x_{i+1}, run to x_{N-2} X_{N-1}
            elif self.system.bc == "open":
                site_indices = dict()
                for op_index in range(0,np.size(self.model,axis=0)):
                    d=np.size(self.model[op_index])
                    if position + d -1 < self.system.N:
                        site_indices[op_index] = np.zeros(d)
                        for n in range(0,d):
                            site_indices[op_index][n] = position+n
                    else:
                        site_indices[op_index] = None


            #filter these sites for identitys, projectors and operators to act with
            #0 is projector, -1 is identity
            P_indices = dict()
            non_projector_site_indices=dict()
            non_projector_op_indices=dict()

            #throw away keys where site_indices[op_index] = None (OBC)
            op_index_keys = list(site_indices.keys())
            to_del=[]
            for n in range(0,np.size(op_index_keys,axis=0)):
                if site_indices[op_index_keys[n]] is None:
                    to_del = np.append(to_del,n)
            for n in range(np.size(to_del,axis=0)-1,-1,-1):
                op_index_keys = np.delete(op_index_keys,to_del[n],axis=0)
                

            for op_index in op_index_keys:
                for n in range(0,np.size(self.model[op_index],axis=0)):
                    if self.model[op_index][n] != 0 and self.model[op_index][n] != -1: 
                        #init dictionary if empty
                        if op_index not in list(non_projector_site_indices.keys()):
                            non_projector_site_indices[op_index] = np.array([site_indices[op_index][n]])
                            non_projector_op_indices[op_index] = np.array([n])
                        else:
                            non_projector_site_indices[op_index] = np.append(non_projector_site_indices[op_index],site_indices[op_index][n])
                            non_projector_op_indices[op_index] = np.append(non_projector_op_indices[op_index],n)
                    elif self.model[op_index][n] == 0:
                        if op_index not in list(P_indices.keys()):
                            P_indices[op_index] = np.array([site_indices[op_index][n]])
                        else:
                            P_indices[op_index] = np.append(P_indices[op_index],site_indices[op_index][n])

            #loop through all ops in sum to get all the product states mapped to + coef (full H space, sym eq state used later)
            for op_index in op_index_keys:
                #check state survives projectors first:
                survives_projectors = 1
                if op_index in list(P_indices.keys()):
                    for n in range(0,np.size(P_indices[op_index],axis=0)):
                        if state[int(P_indices[op_index][n])] != 0:
                            survives_projectors = 0
                            break
                if survives_projectors == 1:
                    site_indices = non_projector_site_indices[op_index].astype(int)

                    #matrix whos rows are the (on site H space) vectors formed when operator acts on site
                    v = np.zeros((np.size(site_indices),self.system.base),dtype=complex)
                    for n in range(0,np.size(site_indices)):
                        v[n] = self.site_ops[self.model[op_index][int(non_projector_op_indices[op_index][n])]][int(state[int(site_indices[n])])]
                    #seq of tensor products gives all combo of final states due to product of ops Q1 Q2...
                    if np.size(site_indices)>1:
                        x=prod(v[0],v[1],self.system.base)
                        for n in range(2,np.size(site_indices)):
                            x = prod(x,v[n],self.system.base)
                    else:
                        x=v[0]

                    #put this data in 2d array with rows (coef,[c1,c2,c3])
                    #c1,c2,c3... the new bits after operator acted on state (eg |010>->2*|000>, row=2,0)
                    new_bits = np.zeros(np.size(site_indices),dtype=int)
                    coefficients = []
                    for n in range(0,np.size(x,axis=0)):
                        if np.abs(x[n])>0:
                            new_bits = np.vstack((new_bits,int_to_bin_base_m(n,self.system.base,np.size(site_indices)).astype(int)))
                            coefficients = np.append(coefficients,x[n])
                    new_bits = np.delete(new_bits,0,axis=0) #delete init row

                    if np.size(coefficients) !=0:
                        for n in range(0,np.size(new_bits,axis=0)):
                            #form new state bit representation
                            new_state = np.zeros(np.size(state),dtype=int)
                            for m in range(0,np.size(state,axis=0)):
                                if m not in site_indices:
                                    new_state[m] = state[m]
                            if np.size(site_indices)>1:
                                for m in range(0,np.size(site_indices,axis=0)):
                                    new_state[site_indices[m]] = new_bits[n,m]
                            else:
                                new_state[site_indices] = new_bits[n,0]

                            new_ref = bin_to_int_base_m(new_state,self.system.base)
                            if new_ref not in list(new_refs_coef.keys()):
                                if self.uc_size is False:
                                    new_refs_coef[new_ref] = coefficients[n]*self.model_coef[op_index]
                                else:
                                    #check position is correct wrt uc size + site
                                    if position % self.uc_size[op_index] == self.uc_pos[op_index]:
                                        new_refs_coef[new_ref] = coefficients[n]*self.model_coef[op_index]
                            else:
                                if self.uc_size is False:
                                    new_refs_coef[new_ref] = new_refs_coef[new_ref] + coefficients[n]*self.model_coef[op_index]
                                else:
                                    #check position is correct wrt uc size + site
                                    if position % self.uc_size[op_index] == self.uc_pos[op_index]:
                                        new_refs_coef[new_ref] = new_refs_coef[new_ref] + coefficients[n]*self.model_coef[op_index]

        #Use bit maps (refs) mapped from all postions and there coef to update H elements
        all_new_refs = list(new_refs_coef.keys())
        for ref_index in range(0,np.size(all_new_refs,axis=0)):
            if k_vec is None: #if not using syms, simply find location of new state in basis and update H_ij
                if all_new_refs[ref_index] in self.system.basis_refs:
                    new_ref_index = self.system.keys[all_new_refs[ref_index]]
                    self.sector.update_H_entry(i_index,new_ref_index,new_refs_coef[all_new_refs[ref_index]],dim=np.size(block_references))

            else: #must lookup locations in symmetry basis, orbit ref states etc if using symmetries
                if all_new_refs[ref_index] in self.system.basis_refs:
                    #location of state in full basis (to extract periodicty, norm data)
                    j_full_index = self.system.keys[all_new_refs[ref_index]]
                    #sym connected ref state
                    new_ref_sym_conn_ref = self.syms.sym_data[j_full_index,0] 

                    if new_ref_sym_conn_ref in block_references:
                        #location of ref state in sym block basis (for index of Hamiltionian)
                        j_ref_index = block_keys[new_ref_sym_conn_ref]

                        i_ref = block_references[i_index]
                        i_ref_full_index = self.system.keys[i_ref]

                        L = self.syms.sym_data[j_full_index,2:]
                        N_a = self.syms.sym_data[i_ref_full_index,1]
                        N_b = self.syms.sym_data[j_full_index,1]

                        element = N_b/N_a * np.exp(1j * 2*math.pi / self.system.N * np.vdot(k_vec,L))
                        # print(i_index,j_ref_index,np.real(element),new_refs_coef[all_new_refs[ref_index]])
                        self.sector.update_H_entry(i_index,j_ref_index,element*new_refs_coef[all_new_refs[ref_index]],k_vec,dim=np.size(block_references))

    def gen_site_H(self,op_array,coef,site_index,k_vec=None,):
        #construct H by looping through basis and acting on state with each H_i ,H = sum_i (H_i)
        self.sector.transpose(k_vec=k_vec)
        d=np.size(op_array)
        if k_vec is None: #no sym use full basis
            dim = np.size(self.system.basis_refs,axis=0)
            block_references = self.system.basis_refs
        else: #use symmetry block basis
            print("Sector: "+str(k_vec))
            block_references = self.syms.find_block_refs(k_vec)
            dim = np.size(block_references,axis=0)

        print("Dim="+str(dim))
        pbar = ProgressBar()
        #loop through basis states
        for i in pbar(range(0,dim)):
            #set state = bit representation of product state
            state = self.system.basis[self.system.keys[block_references[i]]]
            self.update_H(state,op_array,coef,i,site_index,block_references,k_vec=k_vec)

        self.sector.transpose(k_vec=k_vec)

    #generate H by taking tensor product of single site operators
    def gen_tensor(self):
        if np.size(self.system.unlockers) != self.system.base:
            print("\nConstrained Hilbert space, no tensor product structure, use H.gen() instead of H.gen_tensor()\n")
        else:
            H_gen = np.zeros((self.system.dim,self.system.dim))
            I = np.eye(np.size(self.site_ops[1],axis=0))
            for op_index in range(0,np.size(self.model,axis=0)):
                op_length = np.size(self.model[op_index])
                pbar=ProgressBar()
                for n in pbar(range(0,self.system.N)):
                    op_string = np.zeros(self.system.N)
                    for m in range(0,op_length):
                        index = (n + m) % self.system.N #pbc
                        op_string[index] = self.model[op_index][m]
                    temp = 1
                    for m in range(0,np.size(op_string,axis=0)):
                        if op_string[m] == 0:
                            temp = np.kron(temp,I)
                        else:
                            temp = np.kron(temp,self.site_ops[op_string[m]])
                    H_gen = H_gen + self.model_coef[op_index]*temp
            self.sector.update_H(H_gen)

    def conj(self):
        keys=list(self.sector.table.keys())
        new_H = deepcopy(self)
        for n in range(0,np.size(keys,axis=0)):
            new_H.sector.table[keys[n]].H = np.conj(new_H.sector.table[keys[n]].H)
        return new_H

    def transpose(self):
        keys=list(self.sector.table.keys())
        new_H = deepcopy(self)
        for n in range(0,np.size(keys,axis=0)):
            new_H.sector.table[keys[n]].H = np.transpose(new_H.sector.table[keys[n]].H)
        return new_H

    def herm_conj(self):
        keys=list(self.sector.table.keys())
        new_H = deepcopy(self)
        for n in range(0,np.size(keys,axis=0)):
            new_H.sector.table[keys[n]].H = np.conj(np.transpose(new_H.sector.table[keys[n]].H))
        return new_H

class clock_Hamiltonian(Hamiltonian):
    def clock_matrix(self,k):
        if k % 2 != 0:
            N = np.arange(-(k-1)/2,(k-1)/2+1)
        else:
            N=np.arange(-(k-1)/2,0.5)
            N = np.append(N,-np.flip(N,axis=0))

        eigvalues = np.exp(2*1j *math.pi * N/k)
        eigvectors = np.zeros((k,k),dtype=complex)
        for n in range(0,k):
            for m in range(0,k):
                eigvectors[m,n] = np.power(eigvalues[n],-m)
        eigvectors = 1/np.power(k,0.5)*eigvectors

        outers = dict()
        for n in range(0,k):
            outers[n] = np.zeros((k,k),dtype=complex)
            for i in range(0,np.size(outers[n],axis=0)):
                for j in range(0,np.size(outers[n],axis=1)):
                    outers[n][i,j] = np.conj(eigvectors[i,n]) * eigvectors[j,n]
                    

        C = np.zeros((k,k),dtype=complex)
        for n in range(0,np.size(eigvalues,axis=0)):
            C = C + 2 * 1j* math.pi * N[n] / k * outers[n]

        C = C * -1j
        C = C / np.max(np.abs(np.imag(C)))
        # C = C * 1j * k / (2*math.pi)
        return np.transpose(C)

    def __init__(self,system,syms=None):
        self.system = system
        self.sector = H_table(self.system)
        self.syms=syms

        self.site_ops = dict()
        self.site_ops[1] = self.clock_matrix(self.system.base)
        self.model=np.array([[1]])
        self.model_coef=np.array([1])

        self.uc_size = False
        self.uc_pos = False

class spin_Hamiltonian(Hamiltonian):
    def X(self,s):
        m = [-s]
        temp = -s
        for n in range(0,int(2*s)):
            temp = temp + 1
            m = np.append(m,temp)

        off_diag = np.power(s*(s+1)-m*(m+1),0.5)
        off_diag = off_diag[:np.size(off_diag)-1]

        Sp = np.diag(off_diag,1)
        X = Sp + np.conj(np.transpose(Sp))
        return (X)

    def Y(self,s):
        m = [-s]
        temp = -s
        for n in range(0,int(2*s)):
            temp = temp + 1
            m = np.append(m,temp)

        off_diag = np.power(s*(s+1)-m*(m+1),0.5)
        off_diag = off_diag[:np.size(off_diag)-1]

        Sp = np.diag(off_diag,1)
        Y = -1j*(Sp - np.conj(np.transpose(Sp)))
        return Y

    def Z(self,s):
        m = 2*np.arange(-s,s+1,1)
        Z = np.diag(m)

        return Z

    def __init__(self,system,spin,syms=None):
        self.system = system
        self.sector = H_table(self.system)
        self.syms=syms
        self.spin = spin

        self.uc_size = False
        self.uc_pos = False

        s = 0.5*(self.system.base-1)
        self.site_ops = dict()
        if self.spin == "x":
            self.site_ops[1] = self.X(s)
            # self.site_ops[1] = self.X(s)/2
        elif self.spin == "y":
            self.site_ops[1] = self.Y(s)
            # self.site_ops[1] = self.Y(s)/2
        else:
            self.site_ops[1] = self.Z(s)
            # self.site_ops[1] = self.Z(s)/2
        self.model=np.array([[1]])
        self.model_coef=np.array([1])

class H_entry:
    def __init__(self,H,e=None,u=None):
        self.H = H
        self.e = e
        self.u = u
        self.dim = np.size(H,axis=0)

#lookup table of H in diff sym sectors. No k values defaults to no symmetry
#implemented as dict with bin_to_int_base_m hash function for keys
from scipy.sparse import csc_matrix
from scipy.sparse import linalg as sparse_linalg
class H_table:
    def __init__(self,system):
        self.system=system
        self.table = dict()

    def transpose(self,k_vec=None):
        if k_vec is None: 
            self.table["no_sym"].H= np.transpose(self.table["no_sym"].H)
        else:
            key = bin_to_int_base_m(k_vec,self.system.N) #hash function
            self.table[key].H= np.transpose(self.table[key].H)

    def update_H_entry(self,i,j,val,k_vec=None,dim=None):
        if k_vec is None: 
            #init hamiltonian matrix as emtpy
            if "no_sym" not in list(self.table.keys()):
                self.table["no_sym"]= H_entry(np.zeros((dim,dim),dtype=complex))
            self.table["no_sym"].H[i,j] = self.table["no_sym"].H[i,j] + val
        else:
            key = bin_to_int_base_m(k_vec,self.system.N) #hash function
            if key not in list(self.table.keys()):
                self.table[key]= H_entry(np.zeros((dim,dim),dtype=complex))
            self.table[key].H[i,j] = self.table[key].H[i,j] + val

    def update_H(self,H,index=None):
        if index is None: 
            self.table["no_sym"]= H_entry(H)
        else:
            key = bin_to_int_base_m(index,self.system.N) #hash function
            self.table[key]= H_entry(H)

    def matrix(self,index=None):
        if index is None: 
            return self.table["no_sym"].H
        else:
            key = bin_to_int_base_m(index,self.system.N)
            return self.table[key].H

    def eigvalues(self,index=None):
        if index is None: 
            return self.table["no_sym"].e
        else:
            key = bin_to_int_base_m(index,self.system.N)
            return self.table[key].e

    def eigvectors(self,index=None):
        if index is None: 
            return self.table["no_sym"].u
        else:
            key = bin_to_int_base_m(index,self.system.N)
            return self.table[key].u

    def find_eig(self,index=None,k0=None,verbose=False):
        if index is None:
            self.table["no_sym"].e, self.table["no_sym"].u = np.linalg.eigh(self.table["no_sym"].H)
        else:
            key = bin_to_int_base_m(index,self.system.N)
            self.table[key].e, self.table[key].u = np.linalg.eigh(self.table[key].H)
        if verbose is True:
            print("Found Eigenvalues")
