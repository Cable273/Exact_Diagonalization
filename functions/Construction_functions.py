#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import math
from progressbar import ProgressBar
import itertools
import sys
file_dir = '/home/kieran/Desktop/Work/CMP/spin_chain_models/functions/'
sys.path.append(file_dir)

from Search_functions import find_index_bisection,create_hash_table,hash_search

def int_to_bin(number,bits):
    arr = np.array((number))
    m=bits

    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[...,bit_ix] = fetch_bit_func(strs).astype("int8")

    return ret 

def bin_to_int(bin_arr):
    temp = 0
    for n in range(0,np.size(bin_arr,axis=0)):
        temp = temp + np.power(2,n)*bin_arr[np.size(bin_arr)-n-1]
    return int(temp)

def int_to_bin_base_m(number,m,N):
    basem = np.zeros(N)
    remainder = number
    for n in range(0,N):
        i = N-n-1
        new_remainder = remainder % np.power(m,i)
        basem[n] = int(remainder-new_remainder)/np.power(m,i)
        remainder = new_remainder
    return basem

def bin_to_int_base_m(array,m):
    c = 0
    N = np.size(array)
    for n in range(0,np.size(array,axis=0)):
        i =  N - n -1
        c = c + array[n] * np.power(m,i)
    return int(c)

def spin_flip(state,site):
    if state[site] == 1:
        state[site] = 0
    else:
        state[site] = 1
    return state

def cycle_bits(number,base,bits):
    "cycle bits left on a state"
    state = int_to_bin_base_m(number,base,bits)
    L=np.size(state)
    trans_state=np.zeros(np.size(state))
    trans_state[0:L-1] = state[1:L]
    trans_state[L-1] = state[0]
    return int(bin_to_int_base_m(trans_state,base))

def cycle_bits_state(state):
    "cycle bits left on a state"
    L=np.size(state)
    trans_state=np.zeros(np.size(state))
    trans_state[0:L-1] = state[1:L]
    trans_state[L-1] = state[0]
    return trans_state

def jump_sequence(number,base,bits):
    "generates all other states connected by translation"
    seq=[number]
    cycled_state = cycle_bits(number,base,bits)
    while cycled_state != number:
        seq = np.append(seq,cycled_state)
        #only cycle bits if necessary, *2 if not
        # if cycled_state * 2 > np.power(2,bits)-1:
        cycled_state=cycle_bits(cycled_state,base,bits)
        # else:
            # cycled_state = cycled_state * 2
    return seq

def parity(number,bits):
    state = int_to_bin(number,bits)
    return int(bin_to_int(np.flip(state,axis=0)))

def pairwise_bit_sum(state,bc):
    temp = 0
    if bc == "open":
        for i in range(0,np.size(state)-1,1):
            if state[i] == 1:
                state_i = 1/2
            else:
                state_i = -1/2
            if state[i+1] == 1:
                state_i1 = 1/2
            else:
                state_i1 = -1/2
            temp = temp + state_i * state_i1
        return temp
    elif bc == "periodic":
        temp = 0
        for i in range(0,np.size(state)-1,1):
            if state[i] == 0:
                state_m = -1/2
            else:
                state_m = 1/2

            if state[i+1] == 0:
                state_m1 = -1/2
            else:
                state_m1 = 1/2
            temp = temp + state_m * state_m1

        #pbc do end of chains sep
        if state[0] == 0:
            state_0 = -1/2
        else:
            state_0 = 1/2
        if state[int(np.size(state)-1)] == 0:
            state_N = -1/2
        else:
            state_N = 1/2

        temp = temp + state_0 * state_N
    return temp
