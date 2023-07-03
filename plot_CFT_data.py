import numpy as np
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt



def plot_CFT_data(c_list, CFT_data_list,chi,label):

    cm = "jet"


    cmap = plt.get_cmap("jet")


    c_list= np.real(np.array(c_list))
    CFT_data_list= np.real(np.array(CFT_data_list))


    fig, ax = plt.subplots(figsize=(7,6))

    ax.set_title("Ising CFT data by NNR-TNR ($\\chi=$"+str(chi)+ "$ \ \ $)")
    ax.set_xlabel('RG step', fontsize = 16)
    ax.set_ylabel('c,$\\Delta_{\\alpha}$', fontsize = 16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    #ax.grid()

#for i in range(1,26):
    RGstep=np.arange(3,(len(CFT_data_list)+3))

#    plt.plot(RGstep,0.5*np.ones( len(RGstep)), linestyle="--", color='red')
#    plt.plot(RGstep,(1/8)*np.ones( len(RGstep)), linestyle="--", color=  'blue')
#    plt.plot(RGstep,1*np.ones( len(RGstep)), linestyle="--", color='black')
#    plt.plot(RGstep,(1+1/8)*np.ones( len(RGstep)), linestyle="--", color=  'blue')
#    plt.plot(RGstep,(2)*np.ones( len(RGstep)), linestyle="--", color='black')

#    plt.plot(RGstep,(2+1/8)*np.ones( len(RGstep)), linestyle="--", color=  'blue')#
#    plt.plot(RGstep,(3)*np.ones( len(RGstep)), linestyle="--", color='black')
#    plt.plot(RGstep,(3+1/8)*np.ones( len(RGstep)), linestyle="--", color=  'blue')
#    plt.plot(RGstep,(4)*np.ones( len(RGstep)), linestyle="--", color='black')
#    plt.plot(RGstep,(4+1/8)*np.ones( len(RGstep)), linestyle="--", color=  'blue')

    ax = plt.plot( RGstep ,c_list,markersize=0.6, color='red', linestyle="dashed")


    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.get_cmap("tab20").colors)
    for i in range(40):
        ax = plt.plot( RGstep ,CFT_data_list[:,i],markersize=0.6, label = i)


    plt.ylim(0,4.4)
    plt.xlim(3,len(CFT_data_list)+3)
    import os
    dirname = "CFTdata001/"


    filename = dirname + str(label)+".pdf"
    plt.savefig(filename)
