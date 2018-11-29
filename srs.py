import numpy as np


def chol_to_srs(in_chol,M = None):
    """
    Take packed cholesky spit out S and R
    """
    if M is None:
        #infer size
        # M(M+1)/2 = in_chol.shape[0]
        # =>
        M = int(-1+np.round(np.sqrt(1+8*in_chol.shape[0]))/2) #rounding just in case
    #recover the covariance matrix
    tril_indices = np.tril_indices(M)
    chol_e = np.zeros((M,M))
    chol_e[tril_indices[0], tril_indices[1]] = in_chol
    cov_e = np.dot(chol_e,chol_e.T)
    ## SRS decomposition of the result
    S = np.sqrt(np.diag(cov_e)) #only diagonal elements
    R = np.diag(1/S).dot(cov_e.dot(np.diag(1/S)))
    return (S,R)

def srs_packed_to_srs(Sp,Rp):
    S = np.diag(Sp)
    M = Sp.shape[0]
    Rind = np.triu_indices(M,1)
    R = np.eye(M)*0.5
    R[Rind] = Rp[:]
    R = R+R.T
    return (S, R)

def srs_packed_to_cov(Sp,Rp):
    S, R = srs_packed_to_srs(Sp,Rp)
    return srs_to_cov(S, R)

def srs_to_cov(S, R):
    return S.dot(R.dot(S))

def srs_packed_to_chol(Sp,Rp):
    return np.linalg.cholesky(srs_packed_to_cov(Sp,Rp))

def srs_packed_to_chol_packed(Sp,Rp):
    L = srs_packed_to_chol(Sp,Rp)
    return L[np.tril_indices_from(L)]



def chol_to_srs_packed(in_chol,M = None):
    """
    Take packed cholesky spit out packed srs
    """
    S, R = chol_to_srs(in_chol,M)
    Rupper = R[np.triu_indices_from(R,1)]
    return np.concatenate((S,Rupper))

"""
How apply the above funciton?
-----------------------------
    for i in range(nsamples):
        pos[i,M*K:M*K+(M*(M+1))//2]=chol_to_srs_packed(trace['chol_packed'][i,:])
"""

# This one is used in lse.py
def cov_to_srs_packed(covm):
    (S, R) = cov_to_srs(covm)
    Sp = np.diag(S)
    Rp = R[np.triu_indices_from(R,1)]
    return (Sp,Rp)


def cov_to_srs(covm):
    # decompose the covariance matrix into std and corr
    S = np.diag(np.sqrt(np.diag(covm)))
    inv_S = np.linalg.inv(S)
    R = inv_S.dot(covm.dot(inv_S))
    return (S,R)
