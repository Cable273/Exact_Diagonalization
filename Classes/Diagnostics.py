#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

def check_orthornormal(basis):
    is_orthogonal = 1
    for n in range(0,np.size(basis,axis=1)):
        for m in range(0,np.size(basis,axis=1)):
            temp = np.abs(np.vdot(basis[:,n],basis[:,m]))
            if temp>1e-5:
                print(n,m,temp)
                if n!=m:
                    is_orthogonal = 0
    if is_orthogonal ==1:
        print("Basis orthogonal")
    else:
        print("Basis NOT orthogonal")

def is_hermitian(Q,tol):
    return (np.abs(np.conj(np.transpose(Q))-Q).all())<tol

def is_unitary(U,tol):
    return (np.abs(np.dot(np.conj(np.transpose(U)),U)-np.eye(np.size(U,axis=0))).all())<tol

def print_wf(state,system,tol):
    for n in range(0,np.size(state,axis=0)):
        if np.abs(state[n])>tol:
            print(np.abs(state[n]),system.basis[n])
