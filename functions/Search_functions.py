#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import math
from progressbar import ProgressBar

def find_index_bisection(s,x):
    "find index of array x which contains s using bisection"
    b_min = 1
    b_max = np.size(x)
    breaker=0

    b=0
    if x[0] == s:
        return 0
    elif x[np.size(x)-1] == s:
        return int(np.size(x)-1)
    else:
        while x[b] != s:
            if (b_max - b_min) % 2 ==0:
                b = int(b_min + (b_max - b_min)/2)
                if s < x[b]:
                    b_max = b-1
                elif s > x[b]:
                    b_min = b+1
            else:
                b = int(b_min + (b_max - b_min-1)/2)
                if s < x[b]:
                    b_max = b-1
                elif s > x[b]:
                    b_min = b+1
        return int(b)

def hash(val,hash_table_size):
    return val % hash_table_size

def rehash(pos,hash_table_size):
    return (pos+1) % hash_table_size

def create_hash_table(data,size):
    hash_table = [None] * size
    print("Creating Hash Table")
    pbar = ProgressBar()
    for n in pbar(range(0,np.size(data,axis=0))):
        index = hash(data[n],size)

        #collision resolution
        while hash_table[index] != None:
            index = (index+3) % size
        hash_table[index] = data[n]

    return np.array(hash_table)

def hash_search(val,hash_table):
    index = hash(val,np.size(hash_table))

    count = 0
    break_check = 0
    while hash_table[index] != val:
        if count >= np.size(hash_table):
            break_check = 1
            break
        index = (index+1) % np.size(hash_table)
        count +=1

    if break_check == 0:
        return index
    else:
        return None

