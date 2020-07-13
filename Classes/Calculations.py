#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from progressbar import ProgressBar

import sys
file_dir = '/home/kieran/Desktop/Work/CMP/physics_code/Exact_Diagonalization/functions'
sys.path.append(file_dir)
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m
from Search_functions import find_index_bisection

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

def time_evolve_state(state_energy_basis,eigenvalues,t):
    phases = np.exp(-1j * t * eigenvalues)
    return np.multiply(state_energy_basis,phases)

class level_stats:
    def __init__(self,eigvalues):
        self.e = np.sort(eigvalues)

        #remove degenerate eigenvalues
        degen_indices = []
        for n in range(0,np.size(self.e,axis=0)-1):
            if np.abs(self.e[n+1] - self.e[n])<1e-10:
                degen_indices = np.append(degen_indices,n+1)
        for n in range(np.size(degen_indices,axis=0)-1,-1,-1):
            self.e=np.delete(self.e,degen_indices[n])
        dim = np.size(self.e)
        self.e = self.e[int(dim/3):int(2*dim/3)]
        # self.e = self.e[:int(dim/2)]
        # self.e = self.e[int(5*dim/12):int(7*dim/12)]

        # remove zero modes
        # to_del = []
        # for n in range(0,np.size(self.e,axis=0)):
            # if np.abs(self.e[n])<1e-5:
                # to_del = np.append(to_del,n)
        # for n in range(np.size(to_del,axis=0)-1,-1,-1):
            # self.e = np.delete(self.e,to_del[n])

        self.get_level_ratios()

    def get_level_ratios(self):
        e_diff = np.zeros(np.size(self.e)-1)
        for n in range(0,np.size(e_diff,axis=0)):
            e_diff[n] = self.e[n+1]-self.e[n]

        self.level_ratios = np.zeros(np.size(e_diff)-1)
        for n in range(0,np.size(self.level_ratios,axis=0)):
            # self.level_ratios[n] = np.min(np.array((e_diff[n+1]/e_diff[n],e_diff[n]/e_diff[n+1])))
            e_pair = np.array([e_diff[n],e_diff[n+1]])
            self.level_ratios[n] = np.min(e_pair)/np.max(e_pair)

    def mean(self):
        self.mean_level_ratios = np.mean(self.level_ratios)
        return self.mean_level_ratios

    def plot(self):
        # from scipy import stats
        # density = stats.kde.gaussian_kde(self.level_ratios)
        # x=np.arange(0,5.01,0.01)
        # plt.plot(x,density(x))
        plt.hist(self.level_ratios,normed=True)


#fastest to get init states in energy rep and evolve. Saves exponentiating H more than neccessary
class fidelity:
    def __init__(self,init_state,H,k_refs=None):
        self.init_state = init_state
        self.H = H
        self.k_refs = k_refs

    def eval(self,t_range,overlap_state):
        if self.k_refs is None:
            z_energy = np.conj(self.H.sector.eigvectors()[self.init_state.system.keys[self.init_state.ref],:])
            overlap_energy = np.conj(self.H.sector.eigvectors()[overlap_state.key,:])
            f = np.zeros(np.size(t_range))
            for m in range(0,np.size(f,axis=0)):
                evolved_state = time_evolve_state(z_energy,self.H.sector.eigvalues(),t_range[m])
                f[m] = np.abs(np.vdot(evolved_state,overlap_energy))**2
            return f
        else:
            #sectors init state has non zero overlap with
            k_refs = self.H.syms.find_k_ref(self.init_state.ref)

            #get init state, overlap state sym representation in these sectors
            overlap_sym_states = dict()
            init_sym_states = dict()
            for n in range(0,np.size(k_refs,axis=0)):
                overlap_sym_states[n] = overlap_state.sym_basis(k_refs[n],self.H.syms)
                init_sym_states[n] = self.init_state.sym_basis(k_refs[n],self.H.syms)

            #take overlap with energy eigenstates to get energy rep
            overlap_energy_states = dict()
            init_energy_states = dict()
            for n in range(0,np.size(k_refs,axis=0)):
                overlap_energy_states[n] = np.zeros(np.size(overlap_sym_states[n]),dtype=complex)
                init_energy_states[n] = np.zeros(np.size(init_sym_states[n]),dtype=complex)
                for m in range(0,np.size(overlap_energy_states[n],axis=0)):
                    overlap_energy_states[n][m] = np.conj(np.vdot(overlap_sym_states[n],self.H.sector.eigvectors(k_refs[n])[:,m]))
                    init_energy_states[n][m] = np.conj(np.vdot(init_sym_states[n],self.H.sector.eigvectors(k_refs[n])[:,m]))

            #combine sector data
            e = self.H.sector.eigvalues(k_refs[0])
            init_state_energy = init_energy_states[0]
            overlap_state_energy = overlap_energy_states[0]
            for n in range(1,np.size(k_refs,axis=0)):
                e = np.append(e,self.H.sector.eigvalues(k_refs[n]))
                init_state_energy = np.append(init_state_energy,init_energy_states[n])
                overlap_state_energy = np.append(overlap_state_energy,overlap_energy_states[n])

            #time evolve state and take overlaps
            f = np.zeros(np.size(t_range))
            for n in range(0,np.size(f,axis=0)):
                evolved_state = time_evolve_state(init_state_energy,e,t_range[n])
                f[n] = np.abs(np.vdot(evolved_state,overlap_state_energy))**2
            return f

    def plot(self,t_range,overlap_state,label=None):
        f = self.eval(t_range,overlap_state)
        plt.plot(t_range,f,label=label)
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\vert \langle \phi \vert e^{-iHt} \vert \psi \rangle \vert^2$")
        

class eig_overlap:
    def __init__(self,state,H,k_refs=None):
        self.state = state
        self.H = H
        self.k_refs = k_refs

    def eval(self):
        if self.k_refs is None:
            z_energy = np.conj(self.H.sector.eigvectors()[self.state.key,:])
            overlap = np.log10(np.abs(z_energy)**2)
            return overlap
        else:
            #rotate state to sym_basis
            sym_state = self.state.sym_basis(self.k_refs,self.H.syms)
            #overlap with all eigenstates in k_refs symmetry block
            z_energy = np.zeros(np.size(self.H.sector.eigvalues(self.k_refs)),dtype=complex)
            for n in range(0,np.size(z_energy,axis=0)):
                z_energy[n] = np.conj(np.vdot(sym_state,self.H.sector.eigvectors(self.k_refs)[:,n]))
            overlap = np.log10(np.abs(z_energy)**2)
            return overlap

    def plot(self,tol):
        overlap = self.eval()
        eigs = np.copy(self.H.sector.eigvalues(self.k_refs))
        # to_del=[]
        # for n in range(0,np.size(overlap,axis=0)):
            # if overlap[n] < -8:
                # to_del = np.append(to_del,n)
        # for n in range(np.size(to_del,axis=0)-1,-1,-1):
            # overlap=np.delete(overlap,to_del[n])
            # eigs = np.delete(eigs,to_del[n])
        plt.ylim(bottom=tol)
        plt.xlabel(r"$E$")
        plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
        plt.scatter(eigs,overlap,color="blue")

class entropy:
    def __init__(self,system,k_refs=None):
        self.system = system
        self.k_refs = k_refs

        #bipartite split computational basis, 
        #to construct coefficient matrices for singular values (entanglement spectrum)
        self.N_A = int(np.floor(self.system.N/2))
        self.N_B = int(self.system.N-np.floor(self.system.N/2))
        #split basis
        self.basis_A = self.system.basis[:,:self.N_A]
        self.basis_B = self.system.basis[:,self.N_A:]

        self.basis_A_refs = np.zeros(np.size(self.system.basis,axis=0))
        self.basis_B_refs = np.zeros(np.size(self.system.basis,axis=0))
        for n in range(0,np.size(self.system.basis,axis=0)):
            self.basis_A_refs[n] = bin_to_int_base_m(self.basis_A[n],self.system.base)
            self.basis_B_refs[n] = bin_to_int_base_m(self.basis_B[n],self.system.base)

        self.basis_A_refs_unique = np.unique(self.basis_A_refs)
        self.basis_B_refs_unique = np.unique(self.basis_B_refs)

        self.basis_A_keys = dict()
        self.basis_B_keys = dict()
        for n in range(0,np.size(self.basis_A_refs_unique,axis=0)):
            self.basis_A_keys[self.basis_A_refs_unique[n]] = n
        for n in range(0,np.size(self.basis_B_refs_unique,axis=0)):
            self.basis_B_keys[self.basis_B_refs_unique[n]] = n

    def coef_matrix(self,state):
        import math
        M = np.zeros((np.size(self.basis_A_refs_unique),np.size(self.basis_B_refs_unique)),dtype=complex)
        #coefficients of state form matrix M_ij*state^A_i*state^V_j
        for n in range(0,np.size(state)):
            i,j = int(self.basis_A_refs[n]),int(self.basis_B_refs[n])
            i_index,j_index = self.basis_A_keys[i], self.basis_B_keys[j]

            #fix for when mkl gives convergence error
            # r1=np.random.uniform(1,9)
            # r2=np.random.uniform(0,1)
            M[i_index,j_index] = M[i_index,j_index]+state[n]#+1e-14*r1*np.exp(1j*r2)
        return M

    def eSpectrum(self,state):
        M = self.coef_matrix(state)
        #schmidt decomposition
        U,S,V = sp.linalg.svd(M)
        return np.abs(S)**2
            
    def eval(self,state):
        M = self.coef_matrix(state)
        #schmidt decomposition
        U,S,V = sp.linalg.svd(M)
        #remove zeros (stop error in entropy, log)
        to_del=[]
        for n in range(0,np.size(S,axis=0)):
            if S[n] == 0:
                to_del = np.append(to_del,n)
        for n in range(np.size(to_del,axis=0)-1,-1,-1):
            S=np.delete(S,to_del[n])
            
        #von-neuman entropy
        entropy = -np.vdot(np.abs(S)**2,np.log(np.abs(S)**2))
        return entropy

    def bipartite_density_matrix(self,state):
        M = self.coef_matrix(state)
        return np.dot(np.conj(np.transpose(M)),M)

class site_precession:
    def __init__(self,site_H,system):
        self.H = site_H
        self.e,self.u = np.linalg.eigh(self.H)
        self.system = system

        #projectors
        self.P = dict()
        self.P_energy_basis = dict()
        for n in range(0,self.system.base):
            self.P[n] = np.zeros(self.system.base)
            self.P[n][n] = 1
            self.P[n] = np.diag(self.P[n])
            self.P_energy_basis[n] = np.dot(np.conj(np.transpose(self.u)),np.dot(self.P[n],self.u))

    def eval(self,init_state,t_range):
        z_index = init_state
        z_energy = np.conj(self.u[z_index,:])

        projector_expectations = dict()
        for n in range(0,self.system.base):
            projector_expectations[n] = np.zeros(np.size(t_range),dtype=complex)

        #time evolve state, take projector expectations
        for n in range(0,np.size(t_range,axis=0)):
            evolved_state = time_evolve_state(z_energy,self.e,t_range[n])
            for m in range(0,self.system.base):
                projector_expectations[m][n] = np.vdot(evolved_state,np.dot(self.P_energy_basis[m],evolved_state))

        return projector_expectations

    def plot(self,init_state,t_range):
        projector_expectations = self.eval(init_state,t_range)
        for n in range(0,len(projector_expectations)):
            plt.plot(t_range,projector_expectations[n],label=r"$\vert$"+str(n)+r"$\rangle$")
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\langle P_i(t) \rangle$")
        plt.legend()
    
#with computational basis only
class site_projection:
    def __init__(self,H,site):
        self.H = H
        self.system = self.H.system

        #create projection operator on a given site
        self.P = dict()
        self.P_energy = dict()
        for n in range(0,self.system.base):
            self.P[n] = np.zeros((np.size(self.system.basis_refs),np.size(self.system.basis_refs)))
            self.P_energy[n] = np.zeros((np.size(self.system.basis_refs),np.size(self.system.basis_refs)))

        #create projection ops in comp basis
        for n in range(0,np.size(self.system.basis_refs,axis=0)):
            for base_index in range(0,self.system.base):
                if self.system.basis[n][site] == base_index:
                    self.P[base_index][n,n] = 1

        #rotate projection operator to energy basis
        for n in range(0,len(self.P)):
            self.P_energy[n] = np.dot(np.conj(np.transpose(H.sector.eigvectors())),np.dot(self.P[n],H.sector.eigvectors()))

    def eval(self,init_state,t_range):
        z_index = self.system.keys[init_state.ref]
        z_energy = np.conj(self.H.sector.eigvectors()[z_index,:])

        projector_expectations = dict()
        for n in range(0,self.system.base):
            projector_expectations[n] = np.zeros(np.size(t_range),dtype=complex)

        #time evolve state, take projector expectations
        for n in range(0,np.size(t_range,axis=0)):
            evolved_state = time_evolve_state(z_energy,self.H.sector.eigvalues(),t_range[n])
            for m in range(0,self.system.base):
                projector_expectations[m][n] = np.vdot(evolved_state,np.dot(self.P_energy[m],evolved_state))
        return projector_expectations

    def plot(self,init_state,t_range):
        projector_expectations = self.eval(init_state,t_range)
        for n in range(0,len(projector_expectations)):
            plt.plot(t_range,projector_expectations[n],label=r"$i= $"+str(n))
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\langle P_i(t) \rangle$")
        plt.legend()

import networkx as nx
import random
def plot_adjacency_graph(adjacency_matrix,labels=None,largest_comp=False):
    gr = nx.from_numpy_matrix(adjacency_matrix)

    edges,weights = zip(*nx.get_edge_attributes(gr,'weight').items())
    cmap=plt.cm.Blues
    vmin=min(weights)
    vmax=max(weights)
    pos = nx.spring_layout(gr)

    nx.draw_networkx_nodes(gr,pos,node_color='r',node_size=200,alpha=0.8)
    # nx.draw_networkx_edges(gr,pos,width=3,edge_color = weights,edge_cmap = cmap)
    nx.draw_networkx_edges(gr,pos,width=2)
    pos_higher = {} #offset labels
    # y_off = 0.05
    y_off = 0
    for k, v in pos.items():
        pos_higher[k] = (v[0],v[1]+y_off)
    nx.draw_networkx_labels(gr,pos_higher,labels,font_size = 14,font_color="b")

    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    # sm._A = []
    # plt.colorbar(sm)

class connected_comps:
    def __init__(self,H,k=None):
        self.H = H
        self.system = H.system
        self.k = k
        if k is None:
            self.basis_refs = H.system.basis_refs
            self.keys = self.system.keys
        else:
            block_refs = self.H.syms.find_block_refs(k)
            self.basis_refs = block_refs
            self.keys = dict()
            for n in range(0,np.size(block_refs,axis=0)):
                self.keys[block_refs[n]] = n
                
    def maps_to(self,root_ref):
        temp = self.H.sector.matrix(self.k)[:,self.keys[root_ref]]
        maps_to = []
        for n in range(0,np.size(temp,axis=0)):
            if np.abs(temp[n])>1e-5:
                maps_to = np.append(maps_to,self.basis_refs[n])
        return maps_to
    
    def find_new_roots(self,ref_list,found_list):
        new_list = []
        for n in range(0,np.size(ref_list,axis=0)):
            if ref_list[n] not in found_list:
                new_list = np.append(new_list,ref_list[n])
        return new_list
            
    def find_connected_components(self):
        self.components=dict()
        found=[self.basis_refs[0]]
        self.components[0] = self.basis_refs[0]

        #loop through H, find all states it connects to
        #then loop through these states, find all states they connect to
        #etc until found all states in connected subspace

        root_maps_to = self.maps_to(self.basis_refs[0])
        new_roots = self.find_new_roots(root_maps_to,found)
        self.components[0] = np.append(self.components[0],new_roots)
        found = np.append(found,new_roots)

        while np.size(new_roots)>0:
            new_map = []
            for n in range(0,np.size(new_roots,axis=0)):
                new_map = np.append(new_map,self.maps_to(new_roots[n]))
            new_map = np.unique(np.sort(new_map))
            new_roots = self.find_new_roots(new_map,found)
            self.components[0] = np.append(self.components[0],new_roots)
            found = np.sort(np.append(found,new_roots))

        c=1
        while np.size(found) != self.system.dim:
            for n in range(0,np.size(self.basis_refs,axis=0)):
                if self.basis_refs[n] not in found:
                    new_sector_root = self.basis_refs[n]
                    new_root_index = n
                    break
            self.components[c] = self.basis_refs[self.keys[new_sector_root]]
            found = np.append(found,new_sector_root)

            root_maps_to = self.maps_to(self.basis_refs[new_root_index])
            new_roots = self.find_new_roots(root_maps_to,found)
            self.components[c] = np.append(self.components[c],new_roots)
            found = np.sort(np.append(found,new_roots))

            while np.size(new_roots)>0:
                new_map = []
                for n in range(0,np.size(new_roots,axis=0)):
                    new_map = np.append(new_map,self.maps_to(new_roots[n]))
                new_map = np.unique(np.sort(new_map))
                new_roots = self.find_new_roots(new_map,found)
                self.components[c] = np.append(self.components[c],new_roots)
                found = np.sort(np.append(found,new_roots))
            print("Found:"+str(np.size(found))+"/"+str(self.system.dim))
            c=c+1

        self.comp_sizes = np.zeros(len(self.components))
        for n in range(0,len(self.components)):
            self.comp_sizes[n] = np.size(self.components[n])

    def largest_component(self):
        max_index = np.argmax(self.comp_sizes)
        return self.components[max_index]

class gram_schmidt:
    def __init__(self,basis):
        self.basis = basis
        self.ortho_basis = None

    def project(self,u,v):
        return np.dot(u,v)/np.dot(u,u)*u
    
    def ortho(self):
        self.ortho_basis = np.zeros(np.shape(self.basis),dtype=complex)
        self.ortho_basis[:,0] = self.basis[:,0]
        c=1
        for n in range(1,np.size(self.basis,axis=1)):
            #minus projections
            temp = np.copy(self.basis[:,n])
            for m in range(0,c):
                temp = temp - self.project(self.ortho_basis[:,m],temp)
            if (np.abs(temp)<1e-5).all()==False:
                temp = temp / np.power(np.vdot(temp,temp),0.5)
                self.ortho_basis[:,c] = temp
                c = c+1

        # now normalize/delete zeros
        to_del = []
        for n in range(0,np.size(self.ortho_basis,axis=1)):
            norm = np.abs(np.vdot(self.ortho_basis[:,n],self.ortho_basis[:,n]))
            if (np.abs(self.ortho_basis[:,n])>1e-5).any()==False: 
            # if norm > 1e-5:
                # self.ortho_basis[:,n] = self.ortho_basis[:,n]/np.power(np.vdot(self.ortho_basis[:,n],self.ortho_basis[:,n]),0.5)
            # else:
                to_del = np.append(to_del,n)
        for n in range(np.size(to_del,axis=0)-1,-1,-1):
            self.ortho_basis = np.delete(self.ortho_basis,to_del[n],axis=1)

def get_top_band_indices(e,overlap,N,x0,y0,e_diff=None):
    #identify top band to delete, want evolved dynamics just from second band
    #points closest to (200,200)
    d = np.zeros((np.size(overlap)))
    for n in range(0,np.size(overlap,axis=0)):
        if overlap[n] > -15:
            d[n] = np.power((e[n]-x0)**2+(overlap[n]-y0)**2,0.5)
        else:
            d[n] = 10000
    labels = np.arange(0,np.size(d))
    #N+1 largest vals
    d_sorted,labels_sorted = (list(t) for t in zip(*sorted(zip(d,labels))))
    scar_indices = labels_sorted[0]
    e_found = e[labels_sorted[0]]
    c=1
    for n in range(1,np.size(d_sorted,axis=0)):
        energy_differences = np.abs(e_found - e[labels_sorted[n]]*np.ones(np.size(e_found)))
        if (energy_differences > e_diff).all():
            scar_indices = np.append(scar_indices,labels_sorted[n])
            e_found = np.append(e_found,e[labels_sorted[n]])
            c=c+1
        if c == int(N/2):
            break

    #points closest to (-200,200)
    d = np.zeros((np.size(overlap)))
    for n in range(0,np.size(overlap,axis=0)):
        if overlap[n] > -15:
            d[n] = np.power((e[n]+x0)**2+(overlap[n]-y0)**2,0.5)
        else:
            d[n] = 10000
    labels = np.arange(0,np.size(d))
    #N+1 largest vals
    d_sorted,labels_sorted = (list(t) for t in zip(*sorted(zip(d,labels))))
    scar_indices = np.append(scar_indices,labels_sorted[0])
    e_found = e[labels_sorted[0]]
    c=1
    for n in range(1,np.size(d_sorted,axis=0)):
        energy_differences = np.abs(e_found - e[labels_sorted[n]]*np.ones(np.size(e_found)))
        if (energy_differences > e_diff).all():
            scar_indices = np.append(scar_indices,labels_sorted[n])
            e_found = np.append(e_found,e[labels_sorted[n]])
            c=c+1
        if c == int(N/2):
            break
        

    #identify zero energy state with largest overlap
    max_loc = None
    max_val = -1000
    for n in range(0,np.size(e,axis=0)):
        if np.abs(e[n])<1e-1:
            if overlap[n] > max_val:
                max_val = overlap[n]
                max_loc = n

    # if max_val > -1.5:
    scar_indices = np.append(scar_indices,max_loc)
    min_index = np.argmin(e)
    max_index = np.argmax(e)
    # scar_indices = np.append(scar_indices,min_index)
    # scar_indices = np.append(scar_indices,max_index)
    to_del = []
    for n in range(0,np.size(scar_indices,axis=0)):
        if scar_indices[n] is None:
            to_del = np.append(to_del,n)
    for n in range(np.size(to_del,axis=0)-1,-1,-1):
        scar_indices = np.delete(scar_indices,to_del[n])
    return np.unique(np.sort(scar_indices))

def orth(v,orthonormalBasis):
    newV = np.copy(v)
    if np.size(np.shape(orthonormalBasis))>1:
        for n in range(np.size(orthonormalBasis,axis=0)-1,-1,-1):
            newV = newV - np.vdot(newV,orthonormalBasis[n,:])*orthonormalBasis[n,:]
    else:
        newV = newV - np.vdot(newV,orthonormalBasis)*orthonormalBasis
    if np.abs(np.vdot(newV,newV))<1e-10:
        return None
    else:
        return newV / np.power(np.vdot(newV,newV),0.5)

# def gen_krylov_basis(H,dim,init_state_prod_basis,orth=False):
def gen_krylov_basis(H,init_state_prod_basis,dim):
    psi0 = init_state_prod_basis / np.power(np.vdot(init_state_prod_basis,init_state_prod_basis),0.5)
    kBasis = psi0
    currentState = psi0
    for n in range(0,dim):
        nextState = np.dot(H,currentState)
        nextStateOrth = orth(nextState,kBasis)
        if nextStateOrth is not None:
            kBasis = np.vstack((kBasis,nextStateOrth))
            currentState = nextStateOrth
        else:
            break
    return np.transpose(kBasis)

def mps_uc3_angles2wf(thetas,system):
    def A_up(theta,phi):
        return np.array([[0,1j*np.exp(-1j*phi)],[0,0]])

    def A_down(theta,phi):
        return np.array([[np.cos(theta),0],[np.sin(theta),0]])
    #create MPs
    theta3 = thetas[2]
    theta2 = thetas[1]
    theta1 = thetas[0]

    A_ups = dict()
    A_downs = dict()
    A_ups[0] = A_up(theta1,0)
    A_ups[1] = A_up(theta2,0)
    A_ups[2] = A_up(theta3,0)

    A_downs[0] = A_down(theta1,0)
    A_downs[1] = A_down(theta2,0)
    A_downs[2] = A_down(theta3,0)

    tensors = dict()
    K = 3
    for n in range(0,K):
        tensors[n] = np.zeros((2,np.size(A_ups[0],axis=0),np.size(A_ups[0],axis=1)),dtype=complex)
    tensors[0][0] = A_downs[0]
    tensors[0][1] = A_ups[0]

    tensors[1][0] = A_downs[1]
    tensors[1][1] = A_ups[1]

    tensors[2][0] = A_downs[2]
    tensors[2][1] = A_ups[2]

    from MPS import periodic_MPS
    psi = periodic_MPS(system.N)
    for n in range(0,system.N,1):
        psi.set_entry(n,tensors[int(n%3)],"both")

    wf = np.zeros(system.dim,dtype=complex)
    for n in range(0,np.size(system.basis_refs,axis=0)):
        bits = system.basis[n]
        coef = psi.node[0].tensor[bits[0]]
        for m in range(1,np.size(bits,axis=0)):
            coef = np.dot(coef,psi.node[m].tensor[bits[m]])
        coef = np.trace(coef)
        wf[n] = coef
    return wf

def gen_fsa_basis(Hp,psi0,fsa_dim):
    fsa_basis = psi0
    current_state = fsa_basis
    for n in range(0,fsa_dim):
        next_state = np.dot(Hp,current_state)
        if np.abs(np.vdot(next_state,next_state))>1e-5:
            next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
            fsa_basis = np.vstack((fsa_basis,next_state))
            current_state = next_state
        else:
            break
    fsa_basis = np.transpose(fsa_basis)
    return fsa_basis
