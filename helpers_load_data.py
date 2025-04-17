from numpy import sum
from numpy.random import choice
import numpy as np
from numpy import ma as ma
from qutip import *
import h5py
from types import SimpleNamespace
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm



"load the data with qubit in e, cavity vac"
def load_experimental_data_e_vac(path, threshold, pre_selection=True, flip=None, amplitude=False):
    file = h5py.File(path, 'r')
    data = file["data"]
    x = np.array(data["x"][:, 1])  # np.arange(4,100,1)
    chi = 1.36e-3 * 2 * np.pi
    phase = x
    if not amplitude:
        phase = (4*x) * chi / 2
    data_i = data["I"][:]

    if (flip == None):
        flip = bool(threshold < 0)
    ss_data = np.where(data_i < threshold if (
        not flip) else data_i > threshold, 1, 0)

    if pre_selection:
        m0 = ss_data[:, 0::3]  
        m1 = ss_data[:, 1::3]  
        m2 = ss_data[:, 2::3]


        m1_g = ma.masked_array(m1, mask=m0)
        m2_g = ma.masked_array(m2, mask=m0)
        mx_g = ma.masked_array(m2_g, mask=m1_g)
        mx_g_avg = (1-m1_g.mean(0)) * mx_g.mean(0)

        
        "below is the code to reply sun's question"
        # projection on vac given by e
        mx_e = ma.masked_array(m2_g, mask=np.logical_not(m1_g))
        mx_e_avg_vac = m1_g.mean(0) * mx_e.mean(0)
        

    result = {
        'y': mx_e_avg_vac,
        'x': phase,
        'm1': m1_g, # m1 is an array
        'y_raw': mx_g # y_raw is an arry
    }
    return SimpleNamespace(**result)


"load the data with qubit in g, cavity not in vac"
def load_experimental_data_g_not_vac(path, threshold, pre_selection=True, flip=None, amplitude=False):
    file = h5py.File(path, 'r')
    data = file["data"]
    x = np.array(data["x"][:, 1])  # np.arange(4,100,1)
    chi = 1.36e-3 * 2 * np.pi
    phase = x
    if not amplitude:
        phase = (4*x) * chi / 2
    data_i = data["I"][:]

    if (flip == None):
        flip = bool(threshold < 0)
    ss_data = np.where(data_i < threshold if (
        not flip) else data_i > threshold, 1, 0)

    if pre_selection:
        m0 = ss_data[:, 0::3]  
        m1 = ss_data[:, 1::3]  
        m2 = ss_data[:, 2::3]


        m1_g = ma.masked_array(m1, mask=m0)
        m2_g = ma.masked_array(m2, mask=m0)
        mx_g = ma.masked_array(m2_g, mask=m1_g)

        mx_g_avg = (1-m1_g.mean(0)) * mx_g.mean(0)
        "below is the code to reply sun's question"
        mx_g_avg_not_vac = (1-m1_g.mean(0)) * (1 - mx_g.mean(0))
  


    result = {
        'y': mx_g_avg_not_vac,
        'x': phase,
        'm1': m1_g,
        'y_raw': mx_g
    }
    return SimpleNamespace(**result)


"load the data with qubit in e, cavity not vac"
def load_experimental_data_e_not_vac(path, threshold, pre_selection=True, flip=None, amplitude=False):
    file = h5py.File(path, 'r')
    data = file["data"]
    x = np.array(data["x"][:, 1])  # np.arange(4,100,1)
    chi = 1.36e-3 * 2 * np.pi
    phase = x
    if not amplitude:
        phase = (4*x) * chi / 2
    data_i = data["I"][:]

    if (flip == None):
        flip = bool(threshold < 0)
    ss_data = np.where(data_i < threshold if (
        not flip) else data_i > threshold, 1, 0)

    if pre_selection:
        m0 = ss_data[:, 0::3]  
        m1 = ss_data[:, 1::3]  
        m2 = ss_data[:, 2::3]

        m1_g = ma.masked_array(m1, mask=m0)
        m2_g = ma.masked_array(m2, mask=m0)
        mx_g = ma.masked_array(m2_g, mask=m1_g)
  

        mx_g_avg = (1-m1_g.mean(0)) * mx_g.mean(0)
        "below is the code to reply sun's question"
  
        # projection on not vac given by e
        mx_e_not_vac = ma.masked_array(m2_g, mask=np.logical_not(m1_g))
        mx_e_avg_not_vac =  m1_g.mean(0) * (1- mx_e_not_vac.mean(0))
        
  

    result = {
        'y': mx_e_avg_not_vac,
        'x': phase,
        'm1': m1_g,
        'y_raw': mx_g
    }
    return SimpleNamespace(**result)