# -*- coding: utf-8 -*-

#%% Modules
import equinox as eqx  
import jax
import jax.numpy as jnp

from ML_utilities import vec2tens, rotate4

from jax import config
config.update("jax_enable_x64", True)
    
#%% Eigen_network  
class Eigen_network(eqx.Module):
    layers: list
    

    def __init__(self, layers, *, key, **kwargs):
        super().__init__(**kwargs)
        self.layers = []
        
        assert layers[0] == 2, \
            "The first dimension of \"layers\" must be 2. Which are the number of independent eigenvalues on A2"
                
        assert layers[-1] == 3, \
                    "The last dimension of \"layers\" must be 3 [orthotropic case]. This might change in the future"
        
        for in_width, out_width in zip(layers[:-1], layers[1:]):
            key, subkey = jax.random.split(key)
            self.layers.append(eqx.nn.Linear(in_width, out_width, key=subkey))
                  
    def compute_d1_bounds(self, lambda1, lambda2):
        d1_min = -(lambda1/7) - (lambda2/7) + (1.0/35)
        d1_max = (-5*lambda1**2 + 25*lambda1*lambda2 + lambda1 - 5*lambda2**2 + lambda2)/(35*(lambda1 + lambda2))
        return d1_min, d1_max
    
    def compute_d2_bounds(self, lambda1, lambda2, d1):
        d2_min = (lambda2/7) - (4.0/35)
        d2_max = (-1225*d1*lambda1**2 - 2450*d1*lambda1*lambda2 
                  + 1225*d1*lambda1 - 1225*d1*lambda2**2 + 1400*d1*lambda2
                  - 140*d1 - 175*lambda1**3 + 700*lambda1**2*lambda2 
                  + 210*lambda1**2 + 700*lambda1*lambda2**2 
                  - 780*lambda1*lambda2 - 55*lambda1 - 80*lambda2**2 
                  + 80*lambda2 + 4)/(35*(35*d1 + 5*lambda1 + 35*lambda2**2 - 30*lambda2 - 1) + 1e-15)
        
        return d2_min, d2_max
        
    def compute_d3_bounds(self, lambda1, lambda2, d1, d2):
        d3_min = (lambda1/7) - (4.0/35)
        d3_max = (-1225*d1*d2 - 1225*d1*lambda1**2 - 2450*d1*lambda1*lambda2 
                  + 1400*d1*lambda1 - 1225*d1*lambda2**2 + 1400*d1*lambda2 
                  - 280*d1 - 1225*d2*lambda2**2 + 1050*d2*lambda2 - 105*d2 
                  + 700*lambda1**2*lambda2 - 80*lambda1**2 + 700*lambda1*lambda2**2 
                  - 780*lambda1*lambda2 + 80*lambda1 - 80*lambda2**2 + 80*lambda2 
                  - 8)/(35*(35*d1 + 35*d2 + 35*lambda1**2 - 30*lambda1 + 3) + 1e-15)
        
        return d3_min, d3_max
    
    def build_A4(self, lambda1, lambda2, d1, d2, d3):
        return jnp.array([[-d1 - d2 + 6*lambda1/7 - 3/35, d1 + lambda1/7 + lambda2/7 - 1/35, d2 - lambda2/7 + 4/35, 0, 0, 0], 
                          [d1 + lambda1/7 + lambda2/7 - 1/35, -d1 - d3 + 6*lambda2/7 - 3/35, d3 - lambda1/7 + 4/35, 0, 0, 0], 
                          [d2 - lambda2/7 + 4/35, d3 - lambda1/7 + 4/35, -d2 - d3 - 6*lambda1/7 - 6*lambda2/7 + 27/35, 0, 0, 0], 
                          [0, 0, 0, 2*d3 - 2*lambda1/7 + 8/35, 0, 0], 
                          [0, 0, 0, 0, 2*d2 - 2*lambda2/7 + 8/35, 0], 
                          [0, 0, 0, 0, 0, 2*d1 + 2*lambda1/7 + 2*lambda2/7 - 2/35]])
    
    
    def BU_line_d1_d2_d3(self, lambdas, y):       
        d1_min = -4.0/35
        d1_max = (1.0/35)*(-4+35*lambdas[0] - 35*lambdas[0]*lambdas[0])
        
        d1 = (y[0] - (-1))*((d1_max - d1_min)/(1 - (-1))) + d1_min
        
        jax.lax.select(lambdas[0]==1.0, 
                       d1_min,
                       d1)
        
        d2 = (1.0/35)*(1 - 5*lambdas[0])
        d3 = (1.0/35)*(-4 + 5*lambdas[0])
        return jnp.array([d1,d2,d3])
    
    def get_d1_d2_d3(self, lambdas, y):
        # Compute bounds of d1,d2,d3 
        d1_min, d1_max = self.compute_d1_bounds(lambdas[0], lambdas[1])
        
        # linear map network value from range [a, b] to range [c, d]
        # y = (x - a)*(d - c)/(b - a) + c
        d1 = (y[0] - (-1))*((d1_max - d1_min)/(1 - (-1))) + d1_min
        
        d2_min, d2_max = self.compute_d2_bounds(lambdas[0], lambdas[1], d1)
        d2 = (y[1] - (-1))*((d2_max - d2_min)/(1 - (-1))) + d2_min
        
        d3_min, d3_max = self.compute_d3_bounds(lambdas[0], lambdas[1], d1, d2) 
        d3 = (y[2] - (-1))*((d3_max - d3_min)/(1 - (-1))) + d3_min
    
        return jnp.array([d1,d2,d3])
    
    
    def _check(self, x, min_x, y, d1, d2, d3, lambdas, Av):
      if min_x < -1e-11:
        raise ValueError(f"min_val: {min_x} is negative \n\
                          eig_vals: {x}; sum of eigvals: {jnp.sum(x)} \n\
                          d1: {d1}, d2: {d2}, d3: {d3}, \n\
                          lambdas: {lambdas}; sum of lambdas: {jnp.sum(lambdas)} \n\
                          Av: {Av}")
        # jax.debug.print(f"min_val: {min_x} is negative \n eig_vals: {x} \n d1:{d1}, d2: {d2}, d3:{d3}, lambdas:{lambdas} ")
        # jax.debug.breakpoint()

    def check_min_eigval(self, x, min_x, y, d1, d2, d3, lambdas, Av):
      jax.debug.callback(self._check, x, min_x, y, d1, d2, d3, lambdas, Av)
    
    def check_A4(self, A4, y, d1, d2, d3, lambdas, Av):
        x = jnp.linalg.eigvalsh(A4)
        min_x = jnp.min(x)
        self.check_min_eigval(x, min_x, y, d1, d2, d3, lambdas, Av)
        
    def closest_point_to_triang(self, px, py):
        """ 
            Project point to UBT triangle
            
             Code adapted from:
                Ericson, C. Real-Time Collision Detection (1st ed.).
                pp 137-143, CRC Press, 2005
                https://doi.org/10.1201/b14581
        """
        p = jnp.array([px, py])
        a  = jnp.array([1/2, 1/2])
        b = jnp.array([1/3, 1/3])
        c = jnp.array([1.0, 0.0])
    
        ab = b - a
        ac = c - a
        ap = p - a
        
        # Check if P in vertex region outside A
        d1 = jnp.dot(ab, ap)
        d2 = jnp.dot(ac, ap)
    
        checks0 = (d1 <= 0.0) & (d2 <= 0.0)
        results0 = a
        
        # Check if P in vertex region outside B
        bp = p - b
        d3 = jnp.dot(ab, bp)
        d4 = jnp.dot(ac, bp)
        checks1 = (d3 >= 0.0) & (d4 <= d3)
        results1 = b
        
        # Check if P in edge region of AB, if so return projection of P onto AB
        vc = d1*d4 - d3*d2
        checks2 = (vc <=0.0) & (d1 >= 0.0) & (d3 <= 0.0)
        v = d1/(d1 - d3)
        results2 =  a + v*ab
        
        # Check if P in vertex region outside C
        cp = p - c
        d5 = jnp.dot(ab, cp)
        d6 = jnp.dot(ac, cp)
        checks3 = (d6 >= 0.0) & (d5 <= d6)
        results3 = c
        
        # Check if P in edge region of AC, if so return projection of P onto AC
        vb = d5*d2 - d1*d6
        checks4 = (vb <= 0.0) & (d2 >= 0.0) & (d6 <=0.0)
        w = d2/(d2 - d6)
        results4 = a + w*ac
    
        # Check if P in edge region of BC, if so return projection of P onto BC
        va = d3*d6 - d5*d4
        
        tmp1 = (d4 - d3)
        tmp2 = (d5 - d6)
        checks5 = (va <=0.0) & (tmp1 >= 0.0) & (tmp2>=0.0)
        w = tmp1/(tmp1 + tmp2)
        results5 = b + w*(c-b)
        
        # P inside face region. Compute Q through its barycentric coordinates (u,v,w)
        denom = 1.0/(va + vb + vc)
        v = vb*denom
        w = vc*denom
    
        results6 = a + ab*v + ac*w
        
        checks =jnp.array([checks0, checks1, checks2,
                           checks3, checks4, checks5,
                           True])
        
        results = jnp.array([results0, results1, results2,
                             results3, results4, results5,
                             results6])
        
        index_selection = jnp.argmax(checks)
          
        return results[index_selection]
    

    def __call__(self, Av):
        
        _lambdas, _R = jnp.linalg.eigh(vec2tens(Av)) 
                
        lambdas = self.closest_point_to_triang(_lambdas[2], _lambdas[1])
        
        R = _R[:, [2,1,0]]
                    
        # For each linear layer apply "tanh"
        y = lambdas 
        for layer in self.layers:
            y = jax.nn.tanh(layer(y))

        # check if the point (lambda1, lambda2) is on BU line:        
        # line equation for BU is: y = -x + 1 -> 0 = -x + 1 - y
        check = abs((-lambdas[0] + 1) - lambdas[1]) < 1e-12 
        
        # jax.lax.select with compute both branches of the equation...
        d1, d2, d3 = jax.lax.select(check, 
                                    self.BU_line_d1_d2_d3(lambdas, y),
                                    self.get_d1_d2_d3(lambdas, y)
                                    )

        # Compute A4_hat        
        A4_hat = self.build_A4(lambdas[0], lambdas[1], d1, d2, d3)      
        
        # Check eigenvalues
        self.check_A4(A4_hat, y, d1, d2, d3, _lambdas, Av)
        
        # Rotate it to laboratory axes
        A4 = rotate4(A4_hat, R)
                
        return A4, R @ jnp.diag(jnp.array([lambdas[0], 
                                           lambdas[1], 
                                           1.0 - lambdas[0] - lambdas[1]])) @ R.T
    