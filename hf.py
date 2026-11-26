import numpy as np
import pyscf
import scipy
# electron integrals for h2o
mol_h2o = pyscf.gto.M(atom = 'O 0 0 0; H 0 1 0; H 0 0 1', basis = 'sto3g')
h1e = mol_h2o.intor_symmetric('int1e_kin') + mol_h2o.intor_symmetric('int1e_nuc')
h2e = mol_h2o.intor('int2e_sph')
overlap = mol_h2o.intor_symmetric('int1e_ovlp')
dim = h1e.shape[0]
#print(hcore_ao.shape, eri_4fold_ao.shape)

def get_J_qp(gamma,V,q,p):
    J_qp = 0.0
    for r in range(dim):
        for s in range(r):
            J_qp += gamma[s][r] * V[q][r][p][s]
    return J_qp

def get_J(gamma,V):
    J = np.zeros(h1e.shape)
    for q in range(dim):
        for p in range(q):
            J[q][p] = get_J_qp(gamma,V,q,p)
    return J

def get_K_qp(gamma,V,q,p):
    K_qp = 0.0
    for r in range(dim):
        for s in range(r):
            K_qp += gamma[s][r] * V[q][r][s][p]
    return K_qp

def get_K(gamma,V):
    K = np.zeros(h1e.shape)
    for q in range(dim):
        for p in range(q):
            K[q][p] = get_K_qp(gamma,V,q,p)
    return K

def HF_energy(gamma, H, V):
    en = 0.0
    for p in range(dim):
        for q in range(p):
            J_pq = get_J_qp(gamma,V,p,q)
            K_pq = get_K_qp(gamma,V,p,q)
            en += H[p][q] * gamma[q][p] + 0.5 * ((J_pq-K_pq)*gamma[q][p])
    return en


def SCF(iters=100, tol=10e-6):
    # initial guess/precompute h, S, V in AO basis, do first iteration
    gamma = np.eye(dim) #start with identity
    H=h1e.copy()
    S=overlap.copy()
    V=h2e.copy()
    F = H + get_J(gamma,V) -get_K(gamma,V)
    E, C = scipy.linalg.eigh(F,S)

    
# print(SCF())
