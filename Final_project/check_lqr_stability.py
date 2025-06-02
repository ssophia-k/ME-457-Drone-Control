import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from control.matlab import *  # MATLAB-like control toolbox functionality
import control as ct

from full_state_model_coef import *

# 1. Define your state names in the same order as your A‑matrix
state_names = [
    'pn', 'pe', 'pd',
    'u',  'v',  'w',
    'e0', 'e1', 'e2', 'e3',
    'p',  'q',  'r'
]

# 2. Compute eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(A)


# 4. Plot eigenvalues and annotate with the dominant state
plt.figure()
plt.plot(np.real(eigvals), np.imag(eigvals), 'x', markersize=8)
plt.axhline(0, color='gray',linewidth=0.5)
plt.axvline(0, color='gray',linewidth=0.5)
plt.xlabel('Re')
plt.ylabel('Im')
plt.title('Eigenvalues of the A matrix')
plt.grid(True)
plt.show()


#print(eigvecs)


ctrb = ct.ctrb(A,B) #[B, AB, A^2B, A^3B...A^12B]
print(ctrb.shape) 
rank = np.linalg.matrix_rank(ctrb)
print(rank) #rank = 5, not full rank, so not controllable


# Compute SVD of the controllability matrix
U, s, V = np.linalg.svd(ctrb)
# s is a list of singular values
S = np.diag(s)
r = (s > 1e-12*s[0]).sum() # numberical rank (compare to the first singular value to find effectively zero values)
print(r) #rank = 5, which is good because it matches the rank of the controllability matrix
cntrl_basis = U[:, :5] # matrix where each column is a basis vector for the controllable subspace
print(cntrl_basis)


tol_rel = 1e-10        # a numrerical tolerance that is close to zero
ctrl_idx = []          # indices of controllable modes
unctrl_idx = []        # indices of uncontrollable modes

for k in range(A.shape[0]):
    v  = eigvecs[:, k]
    #if we can project the eigenvector onto the controllable subspace and get back the same vector, then it is controllable
    vp = cntrl_basis @ (cntrl_basis.T.conj() @ v) 
    if np.linalg.norm(v - vp) <= tol_rel * np.linalg.norm(v):
        ctrl_idx.append(k)
    else:
        unctrl_idx.append(k)

print("Controllable modes:")
for k in ctrl_idx:
    print(f"  λ = {eigvals[k]: .6g}")

print("\nUncontrollable modes:")
for k in unctrl_idx:
    print(f"  λ = {eigvals[k]: .6g}")