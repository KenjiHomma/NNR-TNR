import numpy as np
from ncon import ncon
import scipy.linalg as scl
import scipy.integrate as integrate

from NNR_loop_optimization import optimize_aug

from plot_spectrum import plot_spectrum
from plot_CFT_data import plot_CFT_data

def Exact_Free_energy(temperature):
        exact_sol = 0
        def funct_to_integrate(theta1,theta2,beta):
            return np.log((np.cosh(2*beta))**2-np.sinh(2*beta)*(np.cos(theta1)+np.cos(theta2)))

        beta = 1/temperature
        integ = integrate.dblquad(funct_to_integrate,0,np.pi,lambda x: 0, lambda x: np.pi,args=([beta]))[0]
        exact_sol = (-1/beta)*((np.log(2)+(1/(2*np.pi**2))*integ))
        return exact_sol
def Ising_tensor(temp):

    beta = 1/temp
    H_local = np.array([[-1,1],[1,-1]])
    M = np.exp(-beta*H_local)
    delta = np.zeros((2,2,2,2))
    delta[0,0,0,0] = 1.
    delta[1,1,1,1] = 1.
    Msr = scl.sqrtm(M)
    T = ncon([delta,Msr,Msr,Msr,Msr],[[1,2,3,4],[-1,1],[-2,2],[3,-3],[4,-4]])

    Ts = []
    for i in range(4):
        Ts.append(T)
    return Ts


def normalize_T(Ts,g):
    """
    Normalization of 4-index tensors Ts
    """
    for i in range(4):
        Ts[i] = Ts[i]/(g**(1/4))
    return Ts
def LN_renormalization(Ts):
    """
    Renormalizing coarse-grained tensors into new one tensors in square lattice using Lavin-Nave TRG method.
    Arguments:
        Ts: Sets of 3-index tensors 
    Output:
       res_Ts:  Sets of 4-index tensors
    """  
    res_Ts= []

    T1 = ncon([Ts[7],Ts[4]],[[-1,1,-3],[-4,1,-2]])
    T2 = ncon([Ts[3],Ts[0]],[[-3,1,-2],[-1,1,-4]])
    res_T0 = ncon([T1,T2],[[-1,-2,1,2],[1,2,-3,-4]])

    T1 = ncon([Ts[1],Ts[6]],[[-1,1,-3],[-4,1,-2]])
    T2 = ncon([Ts[5],Ts[2]],[[-3,1,-2],[-1,1,-4]])
    res_T1 = ncon([T1,T2],[[-1,-2,1,2],[1,2,-3,-4]])

    res_T2 = res_T0.transpose(2,3,0,1)
    res_T3 = res_T1.transpose(2,3,0,1)

    res_Ts.append(res_T0)
    res_Ts.append(res_T1)
    res_Ts.append(res_T2)
    res_Ts.append(res_T3)
    return res_Ts


def LN_TRG_decomp(Ts,chi):
    """
    Decomposing tensors on square lattice into an octagon configuration using Lavin-Nave TRG method.
    
    Arguments:
        Ts: Sets of 4-index tensors
        chi: Bond dimension
    Output:
        LN_decomp: Sets of 3-index tensors 
    """  
    LN_decomp= []
    for i in range(4):

        size1 = np.shape(Ts[i])
        u,s,v = np.linalg.svd(np.reshape(Ts[i],[Ts[i].shape[0]*Ts[i].shape[1],Ts[i].shape[2]*Ts[i].shape[3]]),full_matrices=False)
        if len(s) >  chi :
            u = u[:,:chi]
            s = s[:chi]
            v = v[:chi,:]
        size2 = np.shape(np.diag(s))
        s1 = u@np.sqrt(np.diag(s))
        s2 = np.sqrt(np.diag(s))@v
        LN_decomp.append(np.reshape(s1,[Ts[i].shape[0],Ts[i].shape[1],len(s)]))
        LN_decomp.append(np.reshape(s2,[len(s),Ts[i].shape[2],Ts[i].shape[3]]))

    return LN_decomp
def transfer_matrix(Ts):
    """
    Extracting CFT data from 2 by 2 transfer matrix using Gu-Wen Method.

    Arguments:
        Ts: Sets of 4-index tensors
      
    Output:
        g: normalization factor  
        central_charge: central charge
        scaling_dims: scaling dimensions
    """   
    M1 = ncon([Ts[0],Ts[3]],[[1,-1,2,-3],[-4,2,-2,1]])
    M2 = ncon([Ts[1],Ts[2]],[[-1,2,-3,1],[1,-4,2,-2]])
    M1 =  M1.reshape(M1.shape[0]*M1.shape[1], M1.shape[2]*M1.shape[3])
    M2 =  M2.reshape(M2.shape[0]*M2.shape[1], M2.shape[2]*M2.shape[3])
    M = M1@M2

    eig , _ =  np.linalg.eig(M)
    eig = -np.sort(-eig)
    g = np.trace(M)

    central_charge = (6/(np.pi))*np.log(eig[0]/(g*g))
    scaling_dims = -(0.5/np.pi)*np.log(eig[1:41]/eig[0])
    return g,central_charge,scaling_dims

def NNM_TNR(Ts, OPT_EPS, loop_iter,RG_I,chi,temp,K,rho,solver_eps):

    G = 1

    exact_f = Exact_Free_energy(temp)
    spectrum_list = []
    f_list = []
    c_list = []

    CFT_data_list=[]

    C = 0
    N = 1
    Nplus = 2

    print("\n ===============hyperparameter  ===============\n")
    print("K:", K)
    print("rho:", rho)
    print("chi:", chi)
    print("\n =============== NNR-TNR starts ===============\n")
    for i in range(RG_I):
        print("\n//----------- Renormalization step:   "+ str(i)+ " -----------\n")

        eight_tensors =  LN_TRG_decomp(Ts,chi)
 
        # ----------- NNR loop-optimization ----------- .  
        eight_tensors_p = LN_TRG_decomp(Ts,chi**2)
        eight_tensors = optimize_aug(eight_tensors,eight_tensors_p,loop_iter,K,rho,solver_eps)
        # ----------- NNR loop-optimization ends ----------- .  

        Ts = LN_renormalization(eight_tensors)

        G0 = G
        G,central_c ,scaling_dims = transfer_matrix(Ts)
        Ts = normalize_T(Ts,G)

        C = np.log(G**(1/4))+Nplus*C
        N *= Nplus

        f = -temp*(np.log(G)+2*C)/(2*N)
        print("\n * free energy_error :       " +str(np.abs((f-exact_f)/exact_f))+ "\n")
        print("\n * central charge :       " +str(central_c)+"\n")
        

        ##### Some of physical quantities are stored for plot #####
        T = np.reshape(Ts[0],[Ts[0].shape[0]*Ts[0].shape[1],Ts[0].shape[2]*Ts[0].shape[3]])
        u,spectrum,v = np.linalg.svd(T,full_matrices=False)
        count = 101
        spectrum = spectrum[:count]/spectrum[0]

        if len(spectrum) is not 101:
            size = 101- len(spectrum)
            spectrum = np.hstack((spectrum, np.zeros(size)))
        if len(scaling_dims) is not 40:
            size = 40- len(scaling_dims)
            scaling_dims = np.hstack((scaling_dims, np.zeros(size)))
        if i > 2:
            c_list.append(central_c)
            CFT_data_list.append(np.real(scaling_dims).tolist())
            spectrum_list.append(list(spectrum))

        spectrum_old = spectrum




    print("\n *  relative free density error vs RG step:       " +str(f_list)+"\n")
    print("\n *  central charge vs RG step:       " +str(c_list)+"\n")

    CFT_data_list = np.array(CFT_data_list)

    ## Plot the singular value and scaling dimension spectrum (Fig.6 and Fig.7 in our paper)
    # Please comment out if not necessary.

    label = ("NNR_chi="+str(chi)+"K="+str(K)+"rho="+str(rho)+"solver_eps="+str(solver_eps)+"loop_iter"+str(loop_iter)+"aug_lag"+"temp="+str(temp/(2/np.log(1+np.sqrt(2)))))
    plot_spectrum(spectrum_list,chi,label)
    plot_CFT_data(c_list, CFT_data_list,chi,label)


    return Ts
import argparse
"""
    A sample implementation of NNR-TNR for 2D square lattice classical Ising model. 

    This code can be used to reproduce the results in our paper .
    Note that no symmetries (C4 rotational or Z2 internal symmetries) are imposed in this implementation)
    Below, we list several hyper-parameters for users to tune. 

    Parameters:
    chi :  Bond dimension
    temp_ratio : Temperature ratio T/Tc, where Tc refers to the exact transition temperature of 2D classical Ising model.
    RG_step : Number of  RG step.
    OPT_EPS: Stopping threshold for NNR loop optimization.

    Hyper-parameters:
    xi_hyper: the penalty parameter introduced in our paper. If one increases xi, the lower-rank solutions are induced.
    rho_hyper: the penalty schedule parameter.

    OPT_MAX_I: Number of maximum (sweep) iterations for NNR loop optimization. 
    solver_eps: Cut-off ratio for small singular values in the linear-matrix solver. Generally, the smaller the better.

"""

parser = argparse.ArgumentParser(
        description="Simulation of 2D classical Ising model by NNR-TNR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("chi", type=int,nargs="?", help="Bond dimension",default=16)
parser.add_argument("temp_ratio", type=float,nargs="?",help="temp ratio",default=1)
parser.add_argument("RG_step", type=int,nargs="?",help="RG_step",default=51)

parser.add_argument("xi_hyper", type=float,nargs="?",help="xi_hyper",default= 1E-6)
parser.add_argument("rho_hyper", type=float,nargs="?",help="rho_hyper",default= 0.9)

parser.add_argument("OPT_EPS", type=float,nargs="?",help="OPT_EPS ",default= 1E-15)
parser.add_argument("OPT_MAX_I", type=int,nargs="?",help="OPT_MAX_I",default=30)
parser.add_argument("solver_eps", type=float,nargs="?",help="OPT_MAX_I",default= 1E-12)


args = parser.parse_args()
chi = args.chi
temp_ratio = args.temp_ratio
RG_step = args.RG_step
OPT_EPS = args.OPT_EPS
OPT_MAX_I = args.OPT_MAX_I
xi = args.xi_hyper
rho = args.rho_hyper
solver_eps =  args.solver_eps
temp =  temp_ratio*2/np.log(1+np.sqrt(2))

Ts = Ising_tensor(temp)
_ = NNM_TNR(Ts, OPT_EPS, OPT_MAX_I,RG_step,chi,temp,xi,rho,solver_eps)
