#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import math
from progressbar import ProgressBar

def save_obj(obj, name ):
    import pickle
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    import pickle
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f,encoding='latin1')

