import numpy as np
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt



def plot_CFT_error(CFT_data_err_list,chi,label):

    cm = "jet"


    cmap = plt.get_cmap("jet")



    CFT_data_err_list= np.real(np.array(CFT_data_err_list))


    fig, ax = plt.subplots()

 
    ax.set_ylabel('$\\delta c$,$ \\delta \\Delta_{0,1}$', fontsize = 16)
 #   ax.set_ylabel('$|\\delta  ^{2}-E^{2}_{exact}|/ E^{2}_{exact}$', fontsize = 16)
    plt.yscale('log')
    plt.xscale('log')
    ax.set_xlabel('L', fontsize = 16)
    ax.set_ylim(10**(-8),1) 
    ax.set_xlim(1, 10**6) 
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    #ax.grid()

#for i in range(1,26):
    RGstep=np.arange((len(CFT_data_err_list))) +3
    for i in range(len(RGstep)):
        RGstep[i] = 2**( (RGstep[i]) /2)

    import os
    print(np.shape(RGstep ),np.shape(CFT_data_err_list))
    ax = plt.scatter( RGstep ,CFT_data_err_list[:,0],color='blue',label ='c',marker='x',  s=40,alpha=0.8)
    ax = plt.scatter( RGstep ,CFT_data_err_list[:,1],color='red',label ='$\\Delta_{1}$',marker='x',  s=40,alpha=0.8)
    ax = plt.scatter( RGstep ,CFT_data_err_list[:,2],color='green',label ='$\\Delta_{2}$',marker='x',  s=40,alpha=0.8)
    plt.legend()
    dirname = "CFTdata001/"
    filename = dirname + str(label)+".pdf"
    plt.savefig(filename)

