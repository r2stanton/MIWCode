#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import math
from liblibra_core import * 
from libra_py import *
import numpy as np
import cProfile, pstats, io
import matplotlib.pyplot as plt
from scipy.special import gamma


# In[3]:


def profiler(func):
    # Decorater to profile the code, use for development
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return(retval)
    return(wrapper)

#@profiler
def MIW_Forces(X, params, P = 0, Rsq = 0, Pi = 0, dPijk = 0, ddPijk = 0, dPik = 0, ddPik = 0, gik = 0):
    """
    Args:
        X: Numpy array containing arrays of N world configurations. N x JD dimensions.
        params: Dictionary with simulation parameters
        OPTIONAL: Include matrices of correct shape for force computation *Avoids ~10 allocs per MD step*
    Returns:
        forces: Python list of the MIW forces, for addition to the classical ones in Nuclear object.f prior to propagation.
    """
    # D is dimensionality of space, J is number of particles per world, N is number of worlds.
    # b is kernel width parameter, and ktype is the type of kernel (exp or gaussian for now)
    # M is the matrix of size X.shape containing masses for the particles at the relevant entry M[i][k]
    # Notation in all functions consistent with 10.1103/PhysRevE.97.053311 and 10.1103/PhysRevX.4.041013
    N = len(X) 
    b = params["b"]
    D = params["D"]
    J = params["J"]
    M = params["M"]
    ktype = params["ktype"]
    
    if type(P) == int:
        # Initializes all needed multidims with correct shape
        # i/j loop over worlds i.e 1->N in conventional sense
        # k loops over coords i.e. 1->J*D in conventional sense
        Rsq   = np.zeros( (N, N) )
        P     = np.zeros( (N, N) )
        F     = np.zeros( (N, J*D) )
        Pi    = np.zeros( (N) )
        dPijk = np.zeros( (N, N, J*D) )
        dPik  = np.zeros( (N, J*D) )
        ddPijk= np.zeros( (N, N, J*D) )
        ddPik = np.zeros( (N, J*D) )
        gik   = np.zeros( (N, J*D) )

        
    #Compute Rsq matrix
    Rsq = Compute_Rsquared(X, Rsq)

    #Populate P matrix for P[i][j]
    P = Compute_P(X, Rsq, ktype, b, D, P = P)

    #Populate Pi matrix for Pi[i]
    for i in range(N):
        Pi[i] = Compute_Pi(X, P, i)


    #3 index terms computed together to minimize python looping
    #Populate dPijk matrix for dPijk[i][j][k]
    #Populate ddPijk matrix for ddPijk[i][j][k]
    for i in range(N):
        for k in range(J*D):
            for j in range(N):
                dPijk[i][j][k]  = Compute_dPijk(X, Rsq, i, j, k, b, P, ktype)
                ddPijk[i][j][k] = Compute_ddPijk(X, Rsq, i, j, k, b, P, ktype)

                #2 index terms computed in this same loop to minimize python loops further
                #Populate dPik matrix for dPik[i][k]
                #Populate ddPik matrix for ddPik[i][k]
                if j == range(N)[-1]:
                    #Works because for given i, k, once last j is filled out, 
                    #Pijk holds all j values for a given i, k
                    dPik[i][k]  = Compute_dPik(X, i, k, b, dPijk, ktype)
                    ddPik[i][k] = Compute_ddPik(X, i, k, b, ddPijk, ktype)
                    gik[i][k]   = Compute_gik(X, i, k, b, dPik, Pi, ktype) 
    
    if ktype.lower() == "gaussian":         
        # 1D has significantly easier computation.
        if D == 1:
            # print("Using simplified forces for D = 1")
            for n in range(N):
                for l in range(J*D):
                    F[n][l] = -1 * ( ( 2 * gik[n][l] * ( -( 1.0/(Pi[n]**2) ) * dPik[n][l]**2 + ( 1.0/Pi[n] ) * ddPik[n][l] ) ) )

                    for i in range(N):
                        if i != n:
                            F[n][l] += -1 * 2 * gik[i][l] * ( ( 1.0/(Pi[i]**2) ) * dPijk[i][n][l]*dPik[i][l] 
                                                       - ( 1.0/Pi[i] ) * ddPijk[i][n][l] )
        # General case for D != 1                    
        else:
            for n in range(N):
                for l in range(J*D):
                    F[n][l] = -1 * ( ( 2 * gik[n][l] * ( -( 1.0/(Pi[n]**2) ) * dPik[n][l]**2 + ( 1.0/Pi[n] ) * ddPik[n][l] ) ) )

                    for i in range(N):
                        if i != n:
                            F[n][l] += -1 * 2 * gik[i][l] * ( ( 1.0/(Pi[i]**2) ) * dPijk[i][n][l]*dPik[i][l] 
                                                       - ( 1.0/Pi[i] ) * ddPijk[i][n][l] )
                    for k in range(J*D):
                        if k != l:
                            res = 0
                            for j in range(N):
                                res += ( X[n][k]-X[j][k] ) * dPijk[n][j][l]

                            F[n][l] += -1 * 2 * gik[n][k] * ( -( 1.0/(Pi[n]**2) ) * dPik[n][k]*dPik[n][l] 
                                                            - 2.0/( b**2 * Pi[n] ) * res ) 

                    for i in range(N):
                        if i != n:
                            for k in range(J*D):
                                if k != l:
                                    F[n][l] += -1 * 2 * gik[i][k] * ( ( 1.0/(Pi[i]**2) ) * dPijk[i][n][k] * dPik[i][l]
                                                                    + ( 2.0/(b**2 * Pi[i]) ) * ( X[i][k] - X[n][k] ) * dPijk[i][n][l] )    
    elif ktype.lower() == "exponential": 
        for n in range(N):
            for l in range(J*D):
                #Compute dU/dx_n^l
                F[n][l] += -1 * 2*gik[n][l]*( - ( 1/Pi[n]**2 ) * dPik[n][l]**2 + (1/Pi[n]) * ddPik[n][l] )
                
                for i in range(N):
                    if i != n:
                        F[n][l] += -1 * 2*gik[i][l]*( ( 1/Pi[i]**2 ) * dPijk[i][n][l] * dPik[i][l] - ( 1/Pi[i] ) * ddPijk[i][n][l] )
                for k in range(J*D):
                    if k != l:
                        res = 0
                        for j in range(N):
                            res += ( ( X[n][k]-X[j][k] ) / np.sqrt(Rsq[n][j]) ) * ( ( 1.0/np.sqrt(Rsq[n][j]) ) + ( 1.0/b ) ) * dPijk[n][j][l]             
                                                                         
                        F[n][l] += -1 * 2*gik[n][k]*( -( 1/Pi[n]**2 )  * dPik[n][k] * dPik[n][l] - ( 1/Pi[n] ) * res )
                for i in range(N):
                    if i != n:
                        for k in range(J*D):
                            if k != l:
                                F[n][l] += -1 * 2*gik[i][k]*( ( 1/Pi[i]**2 ) * dPijk[i][n][k] * dPik[i][l] + ( 1/Pi[i] ) * ( ( X[i][k]-X[n][k] )/np.sqrt(Rsq[i][n]) ) * ( ( 1/np.sqrt(Rsq[i][n]) ) + ( 1/b ) ) * dPijk[i][n][l] )
    
    U = Compute_MIW_Potential(X, b, gik, M, ktype)                                                    
    return(F, U)




def compute_P(Q, X, dim, b):
    """
    Args:
        Q ( MATRIX(ndof, 1) ): point of interest in the ndof-dimensional phase space
        X ( MATRIX(ndof, ntraj) ): the replicas of the ndof-dimensional world
        dim ( int ): physical dimensionalty
        b ( float ): parameter
           
    Note:
       ndof = npart * dim, where npart = the number of particles, dim - dimensionality of the problem
              we usually don't care about npart, although we care about dim
       ntraj = the number of worlds
    """

    ndof = X.num_of_rows
    ntraj = X.num_of_cols
    b2 = b*b

    p = 0.0
    for itraj in range(ntraj):
        q = Q - X.col(itraj)
        argg = (q.T() * q).get(0,0)/b2
        p += math.exp(-argg)

    nrm_fact = math.sqrt(math.pi) * b

    p = p / (ntraj * POW( nrm_fact, dim) )
    
    return p


def gaussian_kernel(Q):
    """
    Args:
        Q ( MATRIX(ndof, ntraj) ): coordinates of all worlds in the ndof-dimensional phase space
        dim ( int ): physical dimensionalty of each world
        b ( float ): parameter
           
    Note:
       ndof = npart * dim, where npart = the number of particles, dim - dimensionality of the problem
              we usually don't care about npart, although we care about dim
       ntraj = the number of worlds

       The notation used for indices: i, j - indicate worlds (trajectories),  a, b - indicate DOFs, 
       n, m - words of the particles w.r.t. which the derivatives are taken

    """


    ndof = Q.num_of_rows
    ntraj = Q.num_of_cols    
    b2 = b*b
    frc = (2.0/b2)

    dr = MATRIX(ndof, ntraj*ntraj)            # x_i^a - x_j^a = dr.get(a, i*ntraj+j)

    rho = MATRIX(1, ntraj*ntraj)              # these are called P_ij = rho.get(0, i*ntraj+j) in the paper
    sum_rho = MATRIX(1, ntraj)                # these are sum_j { P_ij }  = P.get(0, i)
    drho = MATRIX(ndof, ntraj*ntraj)          # these are d P_ij / dx_i^a = drho(a, i*ntraj+j)    
    sum_drho = MATRIX(ndof, ntraj)            # sum_j { d P_ij / dx_i^a }
    ddrho = MATRIX(ntraj, ntraj)              # these are sum_a { d P_ij / dx_i^a * dx_i^a }
    sum_ddrho = MATRIX(1, ntraj)              # these are sum_{a, j} { d P_ij / dx_i^a * dx_i^a }


    dP = MATRIX(ntraj*ndof, ntraj)            #  d P_i / dx_n^a = dP.get(n*ndof + a, i) = sum_j { dP_ij / dx_n^a }
    g = MATRIX(ndof, ntraj)                   # these are g_i^a = g.get(a, i) = P'_i^k / P_i


    d2P = MATRIX(ntraj*ndof, ntraj*ndof)      #  d P'_i^a / dx_n^b = d2P.get(i*ndof+a, n*ndof+b)

    F = MATRIX(ndof, ntraj)


    nrm = 1.0 / (ntraj * POW( math.sqrt(math.pi) * b, dim) )


    # Compute dr, rho, drho, ddrho, P
    for i in range(ntraj):

        for j in range(ntraj):

            indx = i * ntraj + j

            rij2 = 0.0
            #=========== dr ===========
            for idof in range(ndof):

                dq = Q.get(idof, i) - Q.get(idof, j)
                dr.set(idof, indx, dq)
                rij2 += dq**2

            #=========== rho, sum_rho ===========
            pij = nrm * math.exp(-rij2/b2)
            rho.set(0, indx, pij)
            sum_rho.add(0, i, pij)

            #=========== drho, sum_drho ===========
            for idof in range(ndof):
                dq = dr.get(idof, indx)
                val = -frc*dq*pij
                drho.set(idof, indx, val)
                sum_drho.add(idof, i, val)


            #=========== ddrho, sum_ddrho ===========
            val = -frc * (1.0 - frc*rij2) * pij
            ddrho.set(i, j,  -frc * (1.0 - frc*rij2) * pij )
            sum_ddrho.add(0, i, val)


    # ============== Compute dP, Eq. A3 ======================
    for i in range(ntraj):      # dP_i/
        for n in range(ntraj):  # /dx_n

            if n==i: 
                for idof in range(ndof):
                    res = sum_drho.get(idof, n)
                    dP.set(n*ndof+idof, i, res)
                
            else:
                for idof in range(ndof):              
                    res = - drho.get(idof, i*ntraj+n)
                    dP.set(n*ndof+idof, i, res)
            



    # ============== Compute d2P, Eq. A4 ======================
    # d2P = MATRIX(ntraj*ndof, ntraj*ndof)      #  d P'_i^a / dx_n^b = d2P.get(i*ndof+a, n*ndof+b)
    for i in range(ntraj):      # dP_i/
        for n in range(ntraj):  # /dx_n

            for a in range(ndof):      # /dx_n^a
                for b in range(ndof):  # /dx_n^b

                    if n==i and a==b: 

                        d2P.set(i*ndof+a, n*ndof+b, sum_ddrho.get() )  #FIXME: COMPLETE this 
                       
                    #FIXME: other cases




def Compute_Rsquared(X, Rsq = 0):
    """
    Args:
        X: Numpy array containing arrays of N world configurations. N x JD dimensions.
        OPTIONAL Rsquared: Pass the last step's Rsquared matrix to avoid allocation cost at every step.
    Returns:
        Rsq: Matrix containing elements r_ij^2. Sum of squared distances of all componenents in the two worlds.
    """
    if type(Rsq) == int:
        Rsq = np.zeros((len(X), len(X)))
    else:
        pass
    
    for i in range(len(Rsq)):
        for j in range(len(Rsq[0])):
            Rsq[i][j] = np.inner(X[i]-X[j], X[i]-X[j])
    return(Rsq)

def  Compute_P(X, Rsq, ktype, b, D, P = 0):
    """
    Args:
        X: Numpy array containing arrays of N world configurations. N x JD dimensions.
        Rsq: Matrix containing elements r_ij^2. Sum of squared distances of all componenents in the two worlds.
        OPTIONAL, P: Passes P from last step to avoid reallocations at every dynamics step.
        ktype: Type of kernel being used, exponential or gaussian
    Returns:
        Rsq: Matrix containing elements r_ij^2. Sum of squared distances of all componenents in the two worlds.
    """
    if type(P) == int:
        P = np.zeros((len(X), len(X)))

    if ktype.lower() == 'gaussian':    
        for i in range(len(P)):
            for j in range(len(P)):
                P[i][j] = Compute_Pij(X, i, j, Rsq, b, D, ktype)
                
    # This is unnecessary since ktype is passed, but I'm just leaving it in for now
    # in case I separated these for some reason not immediately clear.
    elif ktype.lower() == 'exponential':
        for i in range(len(P)):
            for j in range(len(P)):
                P[i][j] = Compute_Pij(X, i, j, Rsq, b, D, ktype)
                
    
    return(P)
    
    
def Compute_Pij(X, i, j, Rsq, b, D, ktype):
    """
    Args:
        X: Numpy array containing arrays of N world configurations. N x JD dimensions.
        i: Index of the world being computed for
        j: Index to be summed over in computation of P_i
        b: Kernel width parameter
        D: Dimensionality of the space (used in cofactors)
        ktype: Kernel type ("gaussian" or "exponential")
        rijsquared: Sum of squared distances between coords of worlds i, j, $r_{ij}^2 = \Sigma_{k}( r_{ij}^k )^2$
    Returns:
        Pij: Density contribution of particle j at the position of particle i (Used for forces)
    """
    N = len(X)
    # Gaussian Kernel Computation
    
    if ktype.lower() == "gaussian":
        return ( 1 / ( N * ( np.sqrt(np.pi) * b )**D )  ) * np.exp( - Rsq[i][j] / (b**2) )
    elif ktype.lower() == "exponential":
        return ( ( gamma(D/2) / ( 2*N * math.factorial(D-1) * (np.sqrt(np.pi) * b )**D )  ) * np.exp(- np.sqrt(Rsq[i][j]) / b )  )
    

    
def Compute_Pi(X, P, i):
    """
    Args:
        X: Numpy array containing arrays of N world configurations. N x JD dimensions.
        P: Promputed matrix with elements computed by Compute_Pij() function. P[i][j] = Compute_Pij(X, i, j, ...)
        i: Index of the world being computed for
    Returns:
        Pi: $\Sigma_{j} P_{ij}$
    """
    N = len(X)
    res = 0
    
    for j in range(N):
        res+= P[i][j]
        
    return(res)
        
        
def Compute_dPijk(X, Rsq, i, j, k, b, P,  ktype):
    """
    Args:
        X: Numpy array containing arrays of N world configurations. N x JD dimensions.
        Rsq: Matrix with Rsq[i][j] standard inner product of (X[i][j], X[i][j])
        i: Index of the world being computed for
        j: Index to be summed over in computation of dP_i
        b: Kernel width parameter
        P: Matrix with elements Pij. Precomputed due to their usage in all derivatives.
        ktype: Kernel type ("gaussian" or "exponential")
        Pij: Value for Pij precomputed and stored in the P matrix. Pij = P[i][j]
    """
    
    rij_k = (X[i][k] - X[j][k])
    
    if ktype.lower() == "gaussian":
        return( -(2.0/b**2) * rij_k * P[i][j] )
    elif ktype.lower() == "exponential":
        if i != j:
            return (  -1/b * ( rij_k / np.sqrt(Rsq[i][j]) ) * P[i][j]  ) 
        else:
            return ( 0.0 ) 
    else:
        return("Error, kernel type not yet implemented.")
    
# Remove ktype later
def Compute_dPik(X, i, k, b, dPijk, ktype):
    """
    Args:
        X: Numpy array containing arrays of N world configurations. N x JD dimensions.
        i: Index of the world being computed for
        k: Component within the worlds being computed for.
        b: Kernel width parameter
        P: Matrix with elements Pij. Precomputed due to their usage in all derivatives.
        ktype: Kernel type ("gaussian" or "exponential") 
    Returns:
        res: dP_i^k, Sum over worlds j, of dPijk. Needed for force computation.
    """
    res = 0
    for j in range(len(X)):      
        res += dPijk[i][j][k]

    return(res)

def Compute_ddPijk(X, Rsq, i, j, k, b, P, ktype):
    """
    Args:
        X: Numpy array containing arrays of N world configurations. N x JD dimensions.
        Rsq: Matrix with Rsq[i][j] standard inner product of (X[i][j], X[i][j])
        i: Index of the world being computed for
        j: Index to be summed over in ddP_i
        k: Component within the worlds being computed for.
        b: Kernel width parameter
        P: Matrix with elements Pij. Precomputed due to their usage in all derivatives.
        ktype: Kernel type ("gaussian" or "exponential")
    Returns:
        res: ddP_ij^k. Needed for force computations.
    """
    rij_k = X[i][k] - X[j][k]
    
    if ktype.lower() == "gaussian":
        # Test line
        # return( -2.0/(b**2) * ( 1 - (2.0/(b**2)) * np.sqrt(Rsq[i][j]) ) * P[i][j] )
        
        return( -2.0/(b**2) * ( 1 - (2.0/(b**2)) * rij_k**2 ) * P[i][j] )
        
    elif ktype.lower() == "exponential":
        if i != j:
            return( -1/(np.sqrt(Rsq[i][j]) * b) * ( 1.0 - (rij_k**2)/Rsq[i][j] - (1.0/b) * (rij_k**2)/np.sqrt(Rsq[i][j]) ) * P[i][j] ) 
        else:
            return ( 0 )

#Remove ktype later
def Compute_ddPik(X, i, k, b, ddPijk, ktype):
    res = 0
    for j in range(len(X)):
        res += ddPijk[i][j][k]
    
    return(res)
    
def Compute_gik(X, i, k, b, dPik, Pi, ktype):
    """
    Args:
        X: Numpy array containing arrays of N world configurations. N x JD dimensions.
        i: Index of the world being computed for
        k: Component within the worlds being computed for.
        b: Kernel width parameter
        dPik: Matrix with dPik[i][k] elements for gik[i][k] computation
        Pi: Matrix with Pi[i] elements for gik[i][k] computation
        ktype: Kernel type ("gaussian" or "exponential")
    Returns:
        gik[i][k] value needed for computation of MIW forces and potential
    """
    return( .5 * dPik[i][k] / Pi[i] )

# For use in force computation funciton
def Compute_MIW_Potential(X, b, gik, M, ktype):
    """
    Args:
        X: Numpy array containing arrays of N world configurations. N x JD dimensions.
        b: Kernel width parameter
        gik: Matrix containing gik elements as gik[i][k] **If not passed, will be computed using X.
        Mik: Matrix containing mass for the particle corresponding to X[i][k] as Mik[i][k]
        ktype: Kernel type ("gaussian" or "exponential")
    Returns:
        U: Scalar potential for the MIW potential U(X) ***Same as Compute_MIW_Potential2***
    """
    res = 0
    for i in range(len(X)):
        for k in range(len(X[0])):
            res += .5 * 1/M[i][k] * gik[i][k] ** 2
    return(res)


# For use outside of force computation function:
def Compute_MIW_Potential2(X, params):
    """
    Args:
        X: Numpy array containing arrays of N world configurations. N x JD dimensions.
        params: Dictionary with simulation parameters
    Returns:
        U: Scalar potential for the MIW potential U(X) ***Same as Compute_MIW_Potential***
    """
    N = len(X) 
    b = params["b"]
    D = params["D"]
    J = params["J"]
    M = params["M"]
    ktype = params["ktype"]

    # Initializes all needed multidims with correct shape
    # i/j loop over worlds i.e 1->N in conventional sense
    # k loops over coords i.e. 1->J*D in conventional sense
    Rsq   = np.zeros( (N, N) )
    P     = np.zeros( (N, N) )
    F     = np.zeros( (N, J*D) )
    Pi    = np.zeros( (N) )
    dPijk = np.zeros( (N, N, J*D) )
    dPik  = np.zeros( (N, J*D) )
    ddPijk= np.zeros( (N, N, J*D) )
    ddPik = np.zeros( (N, J*D) )
    gik   = np.zeros( (N, J*D) )

        
    #Compute Rsq matrix
    Rsq = Compute_Rsquared(X, Rsq)

    #Populate P matrix for P[i][j]
    P = Compute_P(X, Rsq, ktype, b, D, P=P)

    #Populate Pi matrix for Pi[i]
    for i in range(N):
        Pi[i] = Compute_Pi(X, P, i)
    for i in range(N):
        for k in range(J*D):
            for j in range(N):
                dPijk[i][j][k]  = Compute_dPijk(X, Rsq, i, j, k, b, P, ktype)
                ddPijk[i][j][k] = Compute_ddPijk(X, Rsq, i, j, k, b, P, ktype)

                #2 index terms computed in this same loop to minimize python loops further
                #Populate dPik matrix for dPik[i][k]
                #Populate ddPik matrix for ddPik[i][k]
                if j == range(N)[-1]:
                    #Works because for given i, k, once last j is filled out, 
                    #Pijk holds all j values for a given i, k
                    dPik[i][k]  = Compute_dPik(X, i, k, b, dPijk, ktype)
                    ddPik[i][k] = Compute_ddPik(X, i, k, b, ddPijk, ktype)
                    gik[i][k]   = Compute_gik(X, i, k, b, dPik, Pi, ktype)
    res = 0
    for i in range(len(X)):
        for k in range(len(X[0])):
            res += .5 * 1/M[i][k] * gik[i][k] ** 2
    return(res)

        
def Populate1DUniform(N, spacing):
    X = np.zeros((N, 1))
    for i in range(N):
        X[i][0] = -(N-1)/2.0 * spacing + i*spacing
    return(X)

def Populate3DRandom(N, particles, boxmax):
    """
    Args:
        N: Number of worlds
        particles: Number of particles per work
        boxmax: max distance from origin along all dimensions for uniform sampling
    Returns:
        X: Uniformly sampled from [-boxmax, boxmax] in all 3 dimensions.
    """
    X = np.random.rand(N, 3*particles)
    X -= .5
    X *= boxmax*2
    return(X)


def Write_Frame_To_XYZ(X, D, filename):
    """
    Args:
        X: World configurations
        D: Dimensionality of the space (1,2,3 should work)
        filename: Desired output file name
    Returns:
        Nothing, modifies "filename" by adding the current world configuration to it in .xyz format
        ***Currently just uses a dummy atom X, will be changed to determine based off M matrix in future.
    """
    array = []
    if D == 1:
        for i in X:
            array.append([i[0], 0, 0])
    
        f = open(filename, "a")
        f.write(str(len(X)) + "\n")
        f.write("blank \n")
        for i in array:
            f.write("H" + "    " + str(i[0]) + "    " + str(i[1]) + "    " + str(i[2]) + "\n")
        f.close()
        
    if D == 2:
        for i in X:
            array.append([i[0], i[1], 0])
    
        f = open(filename, "a")
        f.write(str(len(X)) + "\n")
        f.write("blank \n")
        for i in array:
            f.write("H" + "    " + str(i[0]) + "    " + str(i[1]) + "    " + str(i[2]) + "\n")
        f.close()
        
    if D == 3:
        for i in X:
            array.append([i[0], i[1], i[2]])
    
        f = open(filename, "a")
        f.write(str(len(X)) + "\n")
        f.write("blank \n")
        for i in array:
            f.write("H" + "    " + str(i[0]) + "    " + str(i[1]) + "    " + str(i[2]) + "\n")
        f.close()
    return()

#Don't pass filename if you don't want to write to a file
def verlet(X, V, params, dt, steps, filename = 0, method = 0):
    if method == 0:
        F, U = MIW_Forces(X, params)
        for i in range(steps):
            X = X + V * dt + .5 * np.divide(F, M) * dt * dt
            V = V + .5 * np.divide(F, M) * dt
            F, U = MIW_Forces(X, params)
            V = V + .5 * np.divide(F, M) * dt

            if type(filename) != int:
                Write_Frame_To_XYZ(X, 1, filename)
        
    return()

def FiniteDiff(X, params, dx):                    
    Ucurr = Compute_MIW_Potential2(X, params)
    F = np.zeros( X.shape )
    for i1 in range(N):
        for i2 in range(J*D):
            Xtemp = X.copy()
            Xtemp[i1][i2] += dx                        
            Utemp = Compute_MIW_Potential2(Xtemp, params)
            F[i1][i2] = - (Utemp - Ucurr) / dx
            
    return(F)

def UnitConversions(inp, out, value):
    """
    Args:
        inp: String containing units of "value" as input
        out: String containing desired units for "value" as output
        value: Numerical value in units [inp] for conversion to [out]
    Returns:
        value in units [out]
    """
    #Masses
    if inp.lower() == "amu":
        if out == "au":
            return(value*1836.0)
    elif inp.lower() == "au":
        if out == "amu":
            return(value/1836.0)
    else:
        print("Units not supported")

def InitializeMM(X, params, atomlist = "H"):
    """
    Args:
        X: MIW configuration as numpy array
        atomlist: List of atomic types corresponding to entries of X
        params: Not sure what yet
    Returns:
        Xmm: Returns the N worlds, but as Libra objects containing info such as masses/positions/etc. 
        Ham_mm: Contains information for the computation of the classical forces
        Syst_mm: Needed for Force_MM, where systems are originally loaded into.
    """
    #Set up now only for D = 3
    D = params["D"]
    try:
        params_ff = params["params_ff"]
    except:
        print("No FF defined, using default many body LJ with Rcut = 20A")
        params_ff = {"mb_functional":"LJ_Coulomb","R_vdw_on":0.0,"R_vdw_off":20.0 }
    
    FOLDERNAME = "tempsys"
    PREFIX = "world_"
    N = len(X)
    
    ############################################
    #            READ IN THE FILES             #
    ############################################
    
    # Write X to xyz files by world to set up the libra objects.
    # Each world gets an .xyz file: world_#.xyz for generating system later.
    # This is not optimal way to do this time-wise, but I don't think it will matter.
    if atomlist == "H":
        os.system("rm -r %s" % FOLDERNAME)
        os.system("mkdir %s" % FOLDERNAME)
        
        for i in range(N):
            filename = PREFIX + str(i) + ".xyz"
            f = open(FOLDERNAME+"/"+filename, "w+")
            f.write(str(int(len(X[0])/D)) + "\n")
            f.write("\n")
            
            # I think the only time we'd be computed MM forces would be for 3D systems, so just leaving 3D for now.
            if D == 3:
                for j in range(int(len(X[0])/D)):
                    f.write("H" + "    " + str(X[i][j*D]) + "    " + str(X[i][j*D + 1]) + "    " + str(X[i][j*D] + 2) + "\n")
            else:
                print("Need to decide as to if this will work/be necessary for 2D/1D systems")
                return()
                    
            f.close()
    else:
        return("Not yet implemented")
    
    
    ############################################
    #        LOAD IN CHEMDATA + SETUP FFs      #
    ############################################
    
    # Populate Xmm, the list of libra objects containing the worlds for force computations.
    # I.e. Xmm = [world1, world2, ... , worldN] (indexed from 0)
        
    # Imports relevant chemical data. Must have elements.dat/uff.dat in working dir.
    U = Universe()
    verbose = 0
    LoadPT.Load_PT(U, "elements.dat", verbose)
    
    # Set up the FF interactions.
    uff = ForceField(params_ff)

    
    LoadUFF.Load_UFF(uff)
    verb = 0
    assign_rings = 0
    
    
    
    #Initialize lists which will contain return objects. 
    X_mm = [] # Contains coordinates of worlds in libra objects. X[i][j] (MIW) = X[i].q[j] (Libra Nuclear obj)
    # These objects actually contain much more than just the configuration coords, but forces, masses, etc. as well.
    
    Ham_mm = [] # Contains interaction info, needed to compute forces
    
    Syst_mm = []
    # Example call for computed forces OUTSIDE of this function for world i
    # Compute_forces(X_mm[i], el, Ham_mm[i], 1) 
    
    for i in range(N):
        
        # Load in the i'th world and put it in an instanced system.
        syst = System();
        workingFile = FOLDERNAME +"/" +  PREFIX + str(i) + ".xyz"
        # print("Grabbing coordinates for world", i, "from", workingFile)
        LoadMolecule.Load_Molecule(U, syst, workingFile, "xyz");
        
        
        
        # For computation of forces between all atoms.
        atlst1 = list(range(1,syst.Number_of_atoms+1))
        
        
        # Set up Hamiltonian type + bind it and the sytem.
        ham = Hamiltonian_Atomistic(1, 3*syst.Number_of_atoms)
        ham.set_Hamiltonian_type("MM")
        ham.set_interactions_for_atoms(syst, atlst1, atlst1, uff, verb, assign_rings)
        ham.set_system(syst)
        ham.compute()
        
        # Set up libra objects
        #el = Electronic(1,0)
        mol = Nuclear(3*syst.Number_of_atoms)
        
        # Extract the coordinates so that MIW functions can compute classical forces.
        syst.extract_atomic_q(mol.q)        
        
        # Extract masses so as to not have to code extra atomID->a.u. mass determinations.
        syst.extract_atomic_mass(mol.mass)
        
        # Append the current Nuclear object to X_mm
        X_mm.append(mol)
        
        # Append the current Hamiltonian_Atomistic object to Ham_mm
        Ham_mm.append(ham)
        
        Syst_mm.append(syst)
        # Commented codeblock showing how to change coords in the Libra object
        # to get it to compute forces properly
        
#         print("mol.f[0]", mol.f[0])
#         compute_forces(mol, el, ham, 1)
#         print("mol.f[0]", mol.f[0])
#         mol.q[0] = 500
#         compute_forces(mol, el, ham, 1)
#         print("mol.f[0]", mol.f[0])
        
    return(X_mm, Ham_mm, Syst_mm)

def UpdateNuclear(X, X_mm):
    """
    Args:
        X: Numpy array of dimension N x JD containing configurations
        X_mm: Python list of N Nuclear objects with JD DoFs each.
    Returns:
        X_mm: Updated to be consistent with coordinates in X.
    """
    for i in range(len(X)):
        for j in range(len(X[0])):
            X_mm[i].q[j] = X[i][j]
    return(X_mm)

def MM_Forces(X_mm, Ham_mm, Syst_mm, F, potential = False):
    """
    Args:
        X_mm: Contains N Nuclear objects with relevant data bing masses, classical forces, and positions.
        Ham_mm: Contains information for the computation of the classical forces
        Syst_mm: Not 100% why this is needed, but it is. Where the worlds are originally loaded in from.
        F: 2D N x JD numpy array for forces.
        potential: Set to true for computation+return of classical potential (summed over all worlds).
    """
    
    
    el = Electronic(1,0)
    
    potRes = 0
    
    os.system("echo beforeComputingF")
    
    for i in range(len(X_mm)):
        compute_forces(X_mm[i], el, Ham_mm[i], 1) 
        
        if potential:
            potRes += compute_potential_energy(X_mm[i], el, Ham_mm[i], 1)
    os.system("echo beforeUpdatingF")
    
    for i in range(len(F)):
        for j in range(len(F[0])):
            F[i][j] += X_mm[i].f[j]
            # print(X_mm[i].f[j])
    
    if potential:
        return(F, potRes)
    else:
        return(F)
                      