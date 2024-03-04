import numpy as np
from numpy.linalg import norm, eig
from typing import Optional
import scipy.linalg 
from compute_gauge_invariant_X2 import dot_HLv_L4

def schmidt(U):
    U ,r = np.linalg.qr(U)     
    return U

def LOBPCG_transfer_L4(Ts, m, MAX_ITR, tol, debug=False):
    """
    A sample implementation of the Locally Optimal Block  Conjugate Gradient (LOBCG) method for NNR-TNR algorithm

    This code returns the eigenvalues and eigenvectors of the $l_{4}$ transfer matrix. 
    The total computational complexity scales as  O(m\chi^{6}), where m is the number of eigenvalues and $\chi$ is bond dimension.
    
    We will add the details later.

    """


    n =Ts[0].shape[1]**4

    X = np.random.randn(n, m)
    

    X = np.array([X[:,i] / norm(X[:,i]) for i in range(m)]).T

    X= schmidt(X)
    AX = np.zeros((n, m),dtype = 'complex_')
    BX = np.zeros((n, m),dtype = 'complex_')
    lambda_ = np.zeros(m)
    
    R = np.zeros((n, m),dtype = 'complex_')
    P = np.zeros((n, m),dtype = 'complex_')
    AV = np.zeros((n, 2*m),dtype = 'complex_')
    BV = np.zeros((n, 2*m),dtype = 'complex_')

    A_small = np.zeros((3*m, 3*m),dtype = 'complex_')
    B_small = np.zeros((3*m, 3*m),dtype = 'complex_')
    V = np.zeros((n, 2*m),dtype = 'complex_')
    
    for i in range(m):
        AX[:,i] = dot_HLv_L4((Ts), (X[:,i]))
        BX[:,i] =  X[:,i]
    
    XAX = np.conjugate(X.T) @ AX
    XBX = np.conjugate(X.T) @ BX
    lambda_small,b_small = np.linalg.eig(XAX)

    X = X @ b_small

    for i in range(m):
        AX[:,i] = dot_HLv_L4((Ts), (X[:,i]))
        BX[:,i] = X[:,i]
        
    for i in range(m):
        R[:,i] = AX[:,i] - BX[:,i]*lambda_small[i]
    
    V[:,0:m] = X
    V[:,m:2*m] = R

    V= schmidt(V)
    for k in range(m):
        V[:,k] = V[:,k] / norm(V[:,k])
    
    X = V[:,0:m]
    R = V[:,m:2*m]
    
    for j in range(2*m):
        AV[:,j] = dot_HLv_L4((Ts), (V[:,j]))
        BV[:,j] = V[:,j]
        
    A_small = np.conjugate(V.T) @ AV
    Bm_small = np.conjugate(V.T) @ BV

    lambda_small,b_small = np.linalg.eig(A_small)
    
    for j in range(m):
        X[:,j] = V @ b_small[:,j]
    AX = np.zeros((n, m))
    BX = np.zeros((n, m))
    for i in range(m):
        AX[:,i] = dot_HLv_L4((Ts), (X[:,i]))
        BX[:,i] =  X[:,i]
    
    R = np.zeros((n, m))
    for j in range(m):
        R[:,j] = AX[:,j] - BX[:,j]*lambda_small[j]
    
    OR = np.zeros((n,2*m))
    OR[:, :m] = 0.0
    OR[:, m:2*m] = R

    P = np.zeros((n,m))
    for i in range(m):
        P[:,i] = np.matmul(OR,b_small[:,i])
    
    V = np.zeros((n,3*m))
    AV = np.zeros((n,3*m))
    BV = np.zeros((n,3*m))

    AX = np.zeros((n,m))
    BX = np.zeros((n,m))

    for iter in range(MAX_ITR):
        V[:, 0:m] = X
        V[:, m:2*m] = R
        V[:, 2*m:3*m] = P
      
        V= schmidt(V)

        for k in range(X.shape[1]):
            V[:,k] = V[:,k] / norm(V[:,k])

        for k in range(X.shape[1]+R.shape[1]+1, V.shape[1]):
            V[:,k] = V[:,k] / norm(V[:,k])
        
        X = V[:, 0:m]   
        R = V[:, m:2*m] 
        P= V[:, 2*m:3*m] 
        
        for j in range(3*m):
            AV[:,j] = dot_HLv_L4(Ts, (V[:,j]))

        for j in range(3*m):
            BV[:,j] =  V[:,j]

        A_small = np.matmul(np.conjugate(V.T), AV)
        Bm_small = np.matmul(np.conjugate(V.T), BV)

        lambda_small, b_small =  np.linalg.eig(A_small)

        X = np.matmul(V, b_small[:, 0:m])
        
        for j in range(m):
    
            AX[:,j] = dot_HLv_L4(Ts, (X[:,j]))
        for j in range(m):
            BX[:,j] =  X[:,j]

        for j in range(m):       
            R[:,j] = AX[:,j] - BX[:,j]*lambda_small[j]
        for ik in range(m):
            V[:, ik] = 0.0 
        
        P = np.matmul(V, b_small[:, :m])

        if iter == 0:
            initial_R = np.max(np.abs(R))

        if debug:
            print(iter, np.max(np.abs(R))/initial_R)
        
        if np.max(np.abs(R))/initial_R < tol:
            if debug:
                print("[OK] converged.")
            lambda_ = lambda_small[0:m]
            return X,lambda_

    if iter == MAX_ITR-1:
        print("[ERROR] LOBPCG NOT CONVERGED.")
    return X, lambda_small[0:m]

