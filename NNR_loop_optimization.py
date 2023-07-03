import numpy as np
from ncon import ncon
from numpy import linalg as LA

def compute_N(X_tr):
    size = len(X_tr)
    XX_dagger = [0 for x in range(size-1)]
    for i in range(1,size):
        XX_dagger[i-1]  = ncon([np.conj(X_tr[i]),X_tr[i]],[[-1,1,-2],[-3,1,-4]])
    gamma = XX_dagger[0]
    for i in range(1,size-1):
        gamma = ncon([gamma,XX_dagger[i]],[[-1,1,-3,2],[1,-2,2,-4]])
    return gamma

def compute_W(X_tr,Xp_tr,ii):

    size = len(X_tr)

    if ii%2 == 1:
        temp = ncon([Xp_tr[1],Xp_tr[2]],[[-1,-2,1],[1,-3,-4]])
        temp_conj = ncon([X_tr[1],X_tr[2]],[[-1,-2,1],[1,-3,-4]])
        XX_temp = ncon([temp,temp_conj],[[-1,2,3,-2],[-3,2,3,-4]])

        for i in range(3,size-1,2):

            temp = ncon([Xp_tr[i],Xp_tr[i+1]],[[-1,-2,1],[1,-3,-4]])
            temp_conj = ncon([X_tr[i],X_tr[i+1]],[[-1,-2,1],[1,-3,-4]])
            XX_temp_new = ncon([temp,temp_conj],[[-1,2,3,-2],[-3,2,3,-4]])
            XX_temp= ncon([XX_temp,XX_temp_new],[[-1,1,-3,2],[1,-2,2,-4]])

        temp_p = ncon([Xp_tr[size-1],Xp_tr[0]],[[-1,-2,1],[1,-3,-4]])
        temp = ncon([XX_temp,temp_p],[[1,2,-3,-4],[2,-1,-2,1]])
        sigma = ncon([temp,X_tr[size-1]],[[2,-2,-3,1],[1,2,-1]])

    if ii%2 == 0:
        temp = ncon([Xp_tr[2],Xp_tr[3]],[[-1,-2,1],[1,-3,-4]])
        temp_conj = ncon([X_tr[2],X_tr[3]],[[-1,-2,1],[1,-3,-4]])
        XX_temp = ncon([temp,temp_conj],[[-1,2,3,-2],[-3,2,3,-4]])

        for i in range(4,size,2):

            temp = ncon([Xp_tr[i],Xp_tr[i+1]],[[-1,-2,1],[1,-3,-4]])
            temp_conj = ncon([X_tr[i],X_tr[i+1]],[[-1,-2,1],[1,-3,-4]])
            XX_temp_new = ncon([temp,temp_conj],[[-1,2,3,-2],[-3,2,3,-4]])
            XX_temp= ncon([XX_temp,XX_temp_new],[[-1,1,-3,2],[1,-2,2,-4]])


        temp_p = ncon([Xp_tr[0],Xp_tr[1]],[[-1,-2,1],[1,-3,-4]])
        temp = ncon([temp_p,XX_temp],[[1,-2,-3,2],[2,1,-1,-4]])
        sigma = ncon([temp,X_tr[1]],[[1,-2,3,-1],[-3,3,1]])

    return sigma

def compute_cost_function(X_tr,Xp_tr):
    """
    Convergence measurements for Loop-optimization 

    Arguments:
        X_tr:   Sets of 8-index tensors in octagon configuration. This corresponds to | S >  in the original paper 
        Xp_tr:  Sets of 8-index tensors in octagon configuration. Note that we have slightly different implementation to make the contraction easier. 
                This sets of "Xp_tr" corresponds to | T  > in the original paper.
    Output:
        error : Convergence measurements

    Following the original paper's convention, the leading cost function is given by,

        f   =   < T | T  > - < T | S > - < S | T > + < S | S >

    Instead of directly monitoring cost function f, the following quantity comes handy to see the convergence, which is given by
        
        error = 1 -  < T | S > < S | T > / < T | T > < S | S >
 
    This latter term is sometimes called "fidelity".
    """
  
    size = len(X_tr)
    temp = ncon([X_tr[0],X_tr[1]],[[-1,-2,1],[1,-3,-4]])
    temp_conj = ncon([Xp_tr[0],Xp_tr[1]],[[-1,-2,1],[1,-3,-4]])
    XX_temp = ncon([temp,temp_conj],[[-1,2,3,-2],[-3,2,3,-4]])


    for i in range(2,size,2):

        temp = ncon([X_tr[i],X_tr[i+1]],[[-1,-2,1],[1,-3,-4]])
        temp_conj = ncon([Xp_tr[i],Xp_tr[i+1]],[[-1,-2,1],[1,-3,-4]])
        XX_temp_new = ncon([temp,temp_conj],[[-1,2,3,-2],[-3,2,3,-4]])
        XX_temp= ncon([XX_temp,XX_temp_new],[[-1,1,-3,2],[1,-2,2,-4]])

    fi = ncon([XX_temp],[1,1,2,2])


    temp = ncon([Xp_tr[0],Xp_tr[1]],[[-1,-2,1],[1,-3,-4]])
    temp_conj = ncon([Xp_tr[0],Xp_tr[1]],[[-1,-2,1],[1,-3,-4]])
    XX_temp = ncon([temp,temp_conj],[[-1,2,3,-2],[-3,2,3,-4]])


    for i in range(2,size,2):

        temp = ncon([Xp_tr[i],Xp_tr[i+1]],[[-1,-2,1],[1,-3,-4]])
        temp_conj = ncon([Xp_tr[i],Xp_tr[i+1]],[[-1,-2,1],[1,-3,-4]])
        XX_temp_new = ncon([temp,temp_conj],[[-1,2,3,-2],[-3,2,3,-4]])
        XX_temp= ncon([XX_temp,XX_temp_new],[[-1,1,-3,2],[1,-2,2,-4]])

    ff = ncon([XX_temp],[1,1,2,2])

    temp = ncon([X_tr[0],X_tr[1]],[[-1,-2,1],[1,-3,-4]])
    temp_conj = ncon([X_tr[0],X_tr[1]],[[-1,-2,1],[1,-3,-4]])
    XX_temp = ncon([temp,temp_conj],[[-1,2,3,-2],[-3,2,3,-4]])

    for i in range(2,size,2):

        temp = ncon([X_tr[i],X_tr[i+1]],[[-1,-2,1],[1,-3,-4]])
        temp_conj = ncon([X_tr[i],X_tr[i+1]],[[-1,-2,1],[1,-3,-4]])
        XX_temp_new = ncon([temp,temp_conj],[[-1,2,3,-2],[-3,2,3,-4]])
        XX_temp= ncon([XX_temp,XX_temp_new],[[-1,1,-3,2],[1,-2,2,-4]])

    ii = ncon([XX_temp],[1,1,2,2])

    fidelity = fi**2/(ii*ff)
    error = abs(1-fidelity)

    return error

def TensPermuteTR(X_tr,i):

    """
    "TensPermuteTR" is a function to manage the sweeping of loop-optimization.

    Arguments:
        X_tr:  Sets of 3-index tensors in octagon configuration (denoted as "S" in the original paper).
        i:  this integer refers to the i-th variational tensor X_tr^{i} to be optimized for the next iteration.

    for example, let us call the sets of variational tensors  {X_tr^{0},X_tr^{1}... X_tr^{7}} forming the octagon configuration. 
    Given i-th variational tensor X_tr^{i}, this function permutes the order of sets of 3-index tensors as {X_tr^{i}, X_tr^{i+1}... X_tr^{7} X_tr^{0}...X_tr^{i-1}}
    """
    d = len(X_tr)
    indx =  np.arange(i,d)
    indx = np.append(indx,np.arange(0,i))
    count = 0
    X_tr_copy = X_tr.copy()
    for i in indx:
        X_tr[count] = X_tr_copy[i]
        count  += 1
    return X_tr


def X_tr_fold(M,shape,i):

    if i == 0:
        M = M.reshape(shape[0],shape[1],shape[2],order='F')
    elif i == 1:
        M = M.reshape(shape[1],shape[0],shape[2],order='F')
        M = M.transpose(1,0,2)
    elif i == 2:
        M = M.reshape(shape[2],shape[0],shape[1],order='F')
        M = M.transpose(1,2,0)
    return M

def X_tr_unfold(M,i):

    shape = np.array(np.shape(M))
    if i == 0:
         M = M.reshape(shape[0],shape[1]*shape[2],order='F')
    elif i == 1:
        M = M.transpose(1,0,2)
        M = M.reshape(shape[1],shape[0]*shape[2],order='F')
    elif i == 2:
        M = M.transpose(2,0,1)
        M = M.reshape(shape[2],shape[0]*shape[1],order='F')
    return M

def initialize_Y_M(X_tr):
    M = []
    Y = []
    N = len(X_tr)
    for i in range(N):
        M.append([0 for x in range(3)])
        Y.append([0 for x in range(3)])
    for i in range(N):
        for j in range(3):
            M[i][j]=np.zeros(np.array(np.shape((X_tr[i]))))
            Y[i][j]= np.zeros(np.array(np.shape((X_tr[i]))))

    M = np.array(M)
    Y = np.array(Y)
    return Y,M

def Msum_fun(M):
    N= M.shape[0]
    Msum_out = [0 for x in range(N)]
    for i in range(N):
        Msum_out[i]= M[i,0]+M[i,2]
    return Msum_out

def NuclearNorm_Proximal_Operator(Z,tau):
    """
    Proximal Operator for nuclear norm regularization introduced in J.-F. Cai, E. J. Candes, and Z. Shen, "A singular value
    thresholding algorithm for matrix completion (2008)",arXiv:0810.3286 [math.OC].

    """
    m = Z.shape[0]
    n = Z.shape[1]
    if 2*m < n :
        AAT = Z@Z.T.conjugate()
        S, Sigma2, D = np.linalg.svd(AAT)
        V = np.sqrt(Sigma2)

        n = sum(V > tau)
        Vp = V[:n]-tau
        Vp = Vp[Vp>0]
        mid = Vp/ V[:n]
        X = S[:, :n] @ np.diag(mid) @ D[:n, :] @ Z

        return X

    elif m > 2*n:
        X = NuclearNorm_Proximal_Operator(Z.T, tau)
        X = X.T
        return X
    S,V,D =np.linalg.svd(Z)
    Sigma2 = np.diag(V**2)

    Vp = V-tau
    n = sum(Vp > 0)

    X = S[:, :n] @ np.diag(V[:n]) @ D[:n, :]
    return  X

def ALS_NNR(X_tr,Xp_tr,Y,M,Xi,i,solver_eps):

    Msum= Msum_fun(M)
    Ysum= Msum_fun(Y)

    N_tens = compute_N(X_tr)
    N_mat =  N_tens.reshape( N_tens.shape[0]* N_tens.shape[1], N_tens.shape[2]* N_tens.shape[3])

    W = compute_W(X_tr,Xp_tr,i)
    W = X_tr_unfold(W,1)

    G = Xi*X_tr_unfold(Msum[i],1)+X_tr_unfold(Ysum[i],1)+W
    N = (N_mat+2*Xi*np.eye(N_mat.shape[0]))
    X  =np.linalg.lstsq(N, G.T, rcond= solver_eps)[0]

    X_tr[0]= X_tr_fold(X.T,np.shape(X_tr[0]),1)

    for j in range(0,3,2):
        Df=X_tr_unfold((X_tr[0]-Xi**(-1)*Y[i,j]),j)
        M[i,j]=X_tr_fold(NuclearNorm_Proximal_Operator(Df,(Xi)),np.shape(X_tr[0]),j)
    return X_tr[0],M,Y

def optimize_aug(X_tr,Xp_tr,loop_iter,Xi,rho,solver_eps):
    """
    NNR loop optimization introduced in our paper.

    Arguments:
        X_tr:   Sets of 3-index tensors in octagon configuration. This corresponds to | S  >  in the original paper 
        Xp_tr:  Sets of 3-index tensors in octagon configuration. Note that we have slightly different implementation to make the contraction easier. 
                This sets of "Xp_tr" corresponds to | T > in the original paper.
        loop_iter:  Number of maximum (sweep) iterations for loop optimization. 

        Xi:  the penalty parameter introduced in our paper.
        rho: the penalty schedule parameter.
        solver_eps: Cut-off ratio for small singular values in the linear-equation solver. Generally, the smaller the better.

    Output:
        Ts: Updated sets of 8-index tensors in octagon configuration
 
    """
    error = compute_cost_function(X_tr,Xp_tr)
   
    if abs(error) < 1E-15:
        return X_tr

    n = len(X_tr)
    Y,M = initialize_Y_M(X_tr)
    for j in range(loop_iter):
        for i in range(n):
            X_tr = TensPermuteTR(X_tr,i)
            Xp_tr = TensPermuteTR(Xp_tr,i)

            X_tr[0],M,Y = ALS_NNR(X_tr,Xp_tr,Y,M,Xi,i,solver_eps)

            X_tr = TensPermuteTR(X_tr,n-i)
            Xp_tr = TensPermuteTR(Xp_tr,n-i)

            for k in range(0,3,2):
                Y[i,k]=Y[i,k]+(Xi**(1))*(M[i,k]-X_tr[i])
            
        Xi =(Xi*rho)
       
        error = compute_cost_function(X_tr,Xp_tr)

        if (error) < 1E-15 :
            return X_tr

    return X_tr

