# -*- coding: utf-8 -*-

"""   
    This script shows operations in Mandel notation and the derivation of the parametrization for d1, d2 and d3
    
    Literature:
        "Bauer JK, Böhlke T. Variety of fiber orientation tensors. 
        Mathematics and Mechanics of Solids. 2022;27(7):1185-1211  
        doi:10.1177/10812865211057602"
"""

#%% Modules
import numpy as np
import sympy as sym
from itertools import permutations

sym.init_printing(use_latex=False)

#%% Utility functions for Mandel Notation
def to_mandel(A4):
    res = sym.zeros(6)
    
    mu = sym.sqrt(2)
    
    res[0,0]=A4[0,0,0,0]; res[0,1]=A4[0,0,1,1]; res[0,2]=A4[0,0,2,2]; res[0,3]=mu*A4[0,0,1,2]; res[0,4]=mu*A4[0,0,0,2];  res[0,5]=mu*A4[0,0,0,1]; 
    
    res[1,0]=A4[1,1,0,0]; res[1,1]=A4[1,1,1,1]; res[1,2]=A4[1,1,2,2]; res[1,3]=mu*A4[1,1,1,2]; res[1,4]=mu*A4[1,1,0,2];  res[1,5]=mu*A4[1,1,0,1]; 
    
    res[2,0]=A4[2,2,0,0]; res[2,1]=A4[2,2,1,1]; res[2,2]=A4[2,2,2,2]; res[2,3]=mu*A4[2,2,1,2]; res[2,4]=mu*A4[2,2,0,2];  res[2,5]=mu*A4[2,2,0,1]; 
    
    res[3,0]=mu*A4[1,2,0,0]; res[3,1]=mu*A4[1,2,1,1]; res[3,2]=mu*A4[1,2,2,2]; res[3,3]=2*A4[1,2,1,2]; res[3,4]=2*A4[1,2,0,2];  res[3,5]=2*A4[1,2,0,1]; 
    
    res[4,0]=mu*A4[0,2,0,0]; res[4,1]=mu*A4[0,2,1,1]; res[4,2]=mu*A4[0,2,2,2]; res[4,3]=2*A4[0,2,1,2]; res[4,4]=2*A4[0,2,0,2];  res[4,5]=2*A4[0,2,0,1]; 
    
    res[5,0]=mu*A4[0,1,0,0]; res[5,1]=mu*A4[0,1,1,1]; res[5,2]=mu*A4[0,1,2,2]; res[5,3]=2*A4[0,1,1,2]; res[5,4]=2*A4[0,1,0,2];  res[5,5]=2*A4[0,1,0,1]; 
    
    return res
    

def vec2tens(v):
    f = 1/sym.sqrt(2)
    return sym.Matrix([[v[0],   f*v[5], f*v[4]],
                       [f*v[5],   v[1], f*v[3]],
                       [f*v[4], f*v[3],   v[2]]])

def tens2vec(t):
    mu = sym.sqrt(2)
    return sym.Matrix([   t[0,0],    t[1,1],    t[2,2], 
                       mu*t[1,2], mu*t[0,2], mu*t[0,1]])

def rotate4(A4, Q):
    """
        Code based on: https://github.com/charlestucker3/Fiber-Orientation-Tools/blob/main/Fiber-Orientation-Tools/rotate4.m
        
        Literature: 
            Nadaud and Ferrari, 
            Invariant Tensor-to-Matrix Mappings for Evaluation of Tensorial Expressions,
            Journal of Elasticity 52: 43–61, 1998 
    """
    mu = sym.sqrt(2)
    f = 2/mu
    
    Q4a = sym.Matrix([[Q[0,0]*Q[0,0], Q[0,1]*Q[0,1], Q[0,2]*Q[0,2],  f*Q[0,1]*Q[0,2], f*Q[0,2]*Q[0,0], f*Q[0,0]*Q[0,1]],
                      [Q[1,0]*Q[1,0], Q[1,1]*Q[1,1], Q[1,2]*Q[1,2],  f*Q[1,1]*Q[1,2], f*Q[1,2]*Q[1,0], f*Q[1,0]*Q[1,1]],
                      [Q[2,0]*Q[2,0], Q[2,1]*Q[2,1], Q[2,2]*Q[2,2],  f*Q[2,1]*Q[2,2], f*Q[2,2]*Q[2,0], f*Q[2,0]*Q[2,1]],
                     
                      [mu*Q[1,0]*Q[2,0], mu*Q[1,1]*Q[2,1], mu*Q[1,2]*Q[2,2],  Q[2,2]*Q[1,1], Q[2,0]*Q[1,2], Q[2,1]*Q[1,0]],
                      [mu*Q[2,0]*Q[0,0], mu*Q[2,1]*Q[0,1], mu*Q[2,2]*Q[0,2],  Q[0,2]*Q[2,1], Q[0,0]*Q[2,2], Q[0,1]*Q[2,0]],
                      [mu*Q[0,0]*Q[1,0], mu*Q[0,1]*Q[1,1], mu*Q[0,2]*Q[1,2],  Q[1,2]*Q[0,1], Q[1,0]*Q[0,2], Q[1,1]*Q[0,0]]]) 

    
    Q4b = sym.Matrix([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, Q[1,2]*Q[2,1], Q[1,0]*Q[2,2], Q[1,1]*Q[2,0]],
                      [0, 0, 0, Q[2,2]*Q[0,1], Q[2,0]*Q[0,2], Q[2,1]*Q[0,0]],
                      [0, 0, 0, Q[0,2]*Q[1,1], Q[0,0]*Q[1,2], Q[0,1]*Q[1,0]]])
    
    Q4 = Q4a + Q4b
    
    
    A4rot = Q4 @ A4 @ Q4.T

    return A4rot


def sym_A4(T4):   
    store = sym.ImmutableDenseNDimArray.zeros(3,3,3,3)
    for indices in permutations([0,1,2,3], 4):
        store += sym.permutedims(T4, indices)
    return sym.Rational(1, 24)*store
    
def dev_A2(A2):
    A2_iso = sym.Rational(1,3)*sym.eye(3)
    return A2 - A2_iso

def dev_A4(A2, A4):
    I = np.eye(3)
    tmp1 = sym.Rational(6,7)*sym_A4(sym.tensorproduct(A2, I))
    tmp2 = sym.Rational(3,35)*sym_A4(sym.tensorproduct(I, I))
    return A4 - tmp1 + tmp2
    

#%% Mandel notation - sanity check
# # Create a symmetric 2nd-order tensor
# t = np.random.rand(3,3)
# t = 0.5*(t + t.T)
# tv = tens2vec(t)

# # Scalar product
# #   T_{ij} T_{ij} = Tv @ Tv  
# a = np.einsum('ij,ij', t, t)
# b = tv.T @ tv
# print(a-b[0])

# # Dyadic product
# # T_{ij} T_{kl} = Tv @ Tv.T
# T = np.tensordot(t, t, axes=0)
# T_mandel = to_mandel(T)
# print(T_mandel - (tv @ tv.T) )

# # 4th order tensor contracted with 2nd order identity
# I = np.eye(3)
# T_double_I = np.einsum('ijkl,kl->ij', T, I)
# a = T_mandel @ tens2vec(I)
# print(T_double_I - vec2tens(a))

# # 4th order tensor contracted with 2nd order symmetric tensor
# t2 = np.array([1,2,3,2,4,5,3,5,6]).reshape(3,3)
# tv2 = tens2vec(t2)
# T_double_ones = np.einsum('ijkl,kl->ij', T, t2)
# a = T_mandel @ tv2
# print(T_double_ones - vec2tens(a))

# # 4th order tensor contracted with 4th order tensor
# a = np.einsum('ij, ijkl, kl', t2, T, t2)
# b = tv2.T @ T_mandel @ tv2
# print(a-b[0])

# # A_{ijkl} = R_{im} R_{jn} R_{ko} R_{lp} A_{mnop}
# t2 = np.random.rand(3,3)
# t2 = 0.5*(t2 + t2.T)

# res1 =  np.einsum('im,jn,ko,lp,mnop->ijkl', t2, t2, t2, t2, T)
# res2 = rotate4(T_mandel, t2)
# print(np.max(np.abs(to_mandel(res1).evalf()-res2.evalf())))

#%% Orthotropic parametrization
def get_ortho_parametrization():
    d1, d2, d3, d4, d5, d6, d7, d8, d9 = sym.symbols("d1 d2 d3 d4 d5 d6 d7 d8 d9", real = True)
    
    lambda1, lambda2 = sym.symbols("lambda1 lambda2", nonnegative = True)
    
    sqrt_2 = sym.sqrt(2)
    
    F_tric = sym.Matrix([[-(d1 + d2), d1, d2, -sqrt_2*(d4 + d5), sqrt_2*d6, sqrt_2*d8],
                         [d1, -(d1 + d3), d3, sqrt_2*d4, -sqrt_2*(d6 + d7), sqrt_2*d9],
                         [d2, d3, -(d2 + d3), sqrt_2*d5, sqrt_2*d7, -sqrt_2*(d8 + d9)],
                         [-sqrt_2*(d4 + d5), sqrt_2*d4, sqrt_2*d5, 2*d3, -2*(d8 + d9),-2*(d6 + d7)],
                         [sqrt_2*d6, -sqrt_2*(d6 + d7), sqrt_2*d7, -2*(d8 +d9), 2*d2,-2*(d4 + d5)],
                         [sqrt_2*d8, sqrt_2*d9, -sqrt_2*(d8 + d9), -2*(d6 + d7), -2*(d4 + d5), 2*d1]])
    
    F_ortho = F_tric.subs({d4:0, d5:0, d6:0, d7:0, d8:0, d9:0})

    A2_parametrized = sym.diag([lambda1, lambda2, 1-lambda1-lambda2], unpack=True)

    A4_iso = sym.Rational(7, 35)*sym_A4(sym.tensorproduct(sym.eye(3), sym.eye(3)))

    tmp = sym.Rational(6,7)*sym_A4(
                                    sym.tensorproduct(
                                                        dev_A2(A2_parametrized),
                                                        sym.eye(3)
                                                      )
                                    )

    A4_ortho_parametrized = to_mandel(A4_iso) + to_mandel(tmp) + F_ortho
    
    return A4_ortho_parametrized

#%% Bounds
def parametric_d1_d2_d3():
    d1, d2, d3 = sym.symbols("d1 d2 d3", real = True)
    
    lambda1, lambda2 = sym.symbols("lambda1 lambda2", nonnegative = True)
    
    A4 = get_ortho_parametrization()
    
    eig_vals = A4.eigenvals(multiple=True)[::-1]    # first 3 are available
    
    d1_min = sym.solve(eig_vals[0]>0, d1)
    d2_min = sym.solve(eig_vals[1]>0, d2)
    d3_min = sym.solve(eig_vals[2]>0, d3)
    
    sylversters_crit = [sym.det(A4[:i,:i])>0 for i in range(1,7)]
    
    d3_sol = sym.solve(sylversters_crit[2], d3)
    
    d3_lhs = d3_sol.lhs/d3 # divide by d3 to remove it from lhs
    d3_rhs = d3_sol.rhs
    
    d3_sol = sym.simplify(d3_rhs/d3_lhs)
    
    d2_sol = sym.simplify(sym.solve(d3_sol - d3_min.rhs, d2)[0])
    
    d1_sol = sym.simplify(sym.solve(d3_sol.subs({d2:d2_min.rhs}) - d3_min.rhs, d1)[0])
    
    return [d1_min, d1_sol], [d2_min, d2_sol], [d3_min, d3_sol]
 
#%% Main  
 
if __name__ == "__main__":
    # pass
    d1s, d2s, d3s = parametric_d1_d2_d3()
