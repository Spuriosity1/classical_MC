#!/usr/bin/env python3

import sys

import h5py
import argparse
import numpy as np
import re

# Usage: analye_spiral.py
# 
# Inputs a spiral texture and calculates 
# the spiral's sublattice-resolved orientation vectors. 

parser = argparse.ArgumentParser("Analyse spiral texture")
parser.add_argument("fname", help="Name of the .spins.h5 file to read")


args = parser.parse_args()


_TAG_RE = re.compile(r"L=([0-9]+)")

L = _TAG_RE.findall(args.fname)
print(L)
if L is None:
    print("Could not read L")
    sys.exit(1)
else:
    L = int(L[0])

def linear_to_idx3(lin_idx):
    I = [0,0,0]
    I[0] = lin_idx % L
    lin_idx //= L
    I[1] = lin_idx % L
    lin_idx //= L
    I[2] = lin_idx % L

    return tuple( int(i) if i<L//2 else -L+int(i) for i in I)


with h5py.File(args.fname) as f:
    d = np.array(f["/spin_ft"])
    # Expect [component (x,y,z), sublattice, k-index, {re, im}]

    order_vectors = np.empty([3,d.shape[1],2],dtype=np.complex128)

    for sl in range(d.shape[1]):
        print(f"sublattice {sl}")
        for mu, label in enumerate("xyz"):
            S = d[mu,sl,:,0]+1.0j*d[mu, sl, :,1]
            absS=np.abs(S)
            if np.all(absS < 1e-8):
                print(f"\t S{label} = 0")
            else:
                max_n_i = np.argsort(absS)
                maxS = absS[max_n_i[-1]]

                interesting_q = []
                for lin in reversed(max_n_i):
                    if absS[lin]/maxS < 0.99:
                        break
                    interesting_q.append(lin)

                assert len(interesting_q) == 2

                order_vectors[mu, sl, 0] = S[interesting_q[0]]
                order_vectors[mu, sl, 1] = S[interesting_q[1]]

                print(f"S{label} = {[(S[q], linear_to_idx3(q)) for q in interesting_q]}")



sl_avg_order = np.einsum("ijk->ik",order_vectors)
# rotate to minimise z-axis
# two vectors: re[V], im[V]
# orthogonal? Not guaranteed
# re[V] is new x-axis WLOG, vector rejection of im[V] is new y-axis
reV = np.real(sl_avg_order[:,0] + sl_avg_order[:,1])
imV = np.imag(sl_avg_order[:,0] - sl_avg_order[:,1])

reV/=np.linalg.norm(reV)
imV -= reV@imV * reV

imV_norm = np.linalg.norm(imV)
if imV_norm < 1e-10:
    raise ValueError("re[V] and im[V] are (nearly) parallel — can't form orthonormal basis")
imV /= imV_norm

n = np.cross(reV, imV)

print(reV, imV, n)

R = np.vstack([reV, imV, n])

print(R @ (order_vectors[:,:,0] + order_vectors[:,:,1]))
print(-1.0j* R @ (order_vectors[:,:,0] - order_vectors[:,:,1]))


