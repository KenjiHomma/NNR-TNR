import numpy as np

from numba import jit, f8, i8, b1, void
import time
import matplotlib

import matplotlib.pyplot as plt


def plot_spectrum(RGspec,chi,label):

    RGspec = np.array(RGspec)
    print(np.shape(RGspec))
    n = len(RGspec)

    RGstep = np.arange(3,n+3)
    fig, ax = plt.subplots()
    ax.set_title("spectrum by NNM-TNR ($\\chi=$"+str(chi)+ "$ \ \  T=T_{c}$)")
    ax.set_xlabel('RG step', fontsize = 16)
    for i in range(100):
        ax = plt.plot( RGstep ,RGspec[:,i],markersize=10.0,  color='blue', alpha=0.4)
        ax = plt.scatter( RGstep ,RGspec[:,i], color='blue', alpha=0.4)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.xlim(3,n+3)
    import os
    dirname = "spectrum001/"

    filename = dirname + str(label)+".pdf"
    plt.savefig(filename)
