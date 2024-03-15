# -*- coding: utf-8 -*-

#%% Modules
import pandas as pd
import os 
import glob
from torch.utils.data import Dataset

import numpy as np
import jax.numpy as jnp
import jax

from zipfile import ZipFile

from jax import config
config.update("jax_enable_x64", True)

#%% 
def get_h5_list_of_files(path):
    """
        Return a list with all .h5 files in <path>
    """
    return glob.glob(f'{path}\\*.h5')

class fiber_data(Dataset):
    def __init__(self, path, filter_fn=None, 
                 key=jax.random.PRNGKey(1010), 
                 get_data_up_to = 100, 
                 kwargs=None):
        
        self.path = path            # Folder to look for .h5 files
        self.data = []              # List to store the name of the .h5 files 
        self.filter_fn = filter_fn  # User function to filter data. Must be defined as a function of (t, grad, A, args)
        self.key = key              # Key to generate random numbers
        self.kwargs = kwargs        # Dictionary with keyword arguments for filter_fn
        self.get_data_up_to = get_data_up_to

        # Collect all files in path
        files = get_h5_list_of_files(self.path)
        
        for file in files:
            df = pd.read_hdf(file, key='df')            
            self.data.append(df)
                        
    def __len__(self):
        return len(self.data)   
    
    def __getitem__(self, idx):
        t = self.data[idx]["time"].to_numpy()   # shape = (1000,)
        gradU =  self.data[idx].iloc[0].filter(like='grad', axis=0).to_numpy() # shape = (9,)
                     
        A = self.data[idx].filter(regex='A_{[1-6][1-6]}', axis=1).to_numpy() # shape=(1000,6)
        
        t_idx = np.argmin(np.abs((t - self.get_data_up_to)))
        t =  t[:t_idx]
        A =  A[:t_idx, :]
        
        if self.filter_fn is None:
            return t, gradU, A
        else:
            self.key, subkey = jax.random.split(self.key)
            if self.kwargs is not None:
                return self.filter_fn(t, gradU, A, subkey,**self.kwargs)
            else:
                return self.filter_fn(t, gradU, A, subkey)


#%%  Filter function
def filter_fn(t, gradU, A, key, 
              in_Mandel=False, 
              skip_every=None, 
              n_time_steps=None, 
              n_initial_conditions=None):
    """
        Parameters
        ----------
        t : NUMPY Array 
            Time 
        gradU : NUMPY Array (9,)
            Velocity gradient
        A : NUMPY Array (1000,6)
            2nd-order orientation tensor
        key : JAX PRNGKey
            Key to generate random values
        is_contracted : Bool, optional
            Put *A* in Mandel notation.
        skip_every : Int, optional
            Skip every N steps.
        n_time_steps : Int, optional
            Use n_time_steps.

        Returns
        -------
            t, gradU and A
    """
    
    t_out = t
    A_out = A
    
    if in_Mandel:
        # Original data is organized as: A_xx, A_xy, A_xz, A_yy, A_yz, A_zz
        # Re-organize to be in Mandel notation
        f = np.sqrt(2)
        A_out = np.column_stack([A[:, 0], A[:, 3], A[:, 5], f*A[:, 4], f*A[:, 2], f*A[:, 1]]) 
        
    if skip_every is not None:
        t_out = t[::skip_every]
        A_out = A_out[::skip_every, :]
        
    if n_initial_conditions is not None:
        assert isinstance(n_time_steps, int), " \"n_time_steps\" must be an integer"
        
    if n_time_steps is not None:
        assert isinstance(n_initial_conditions, int), " \"n_initial_conditions\" must be an integer"
        
        # If n_initial_conditions is zero, work as increment of fits
        # data from t=0 to t=n_time_step
        if n_initial_conditions == 0:
            s = [0]
        else:
            s = jax.random.choice(key, jnp.arange(len(t_out) - n_time_steps, dtype=np.int64),
                                  shape=(n_initial_conditions,), 
                                  replace=False)
        
        t_out = np.stack([t_out[idx : idx + n_time_steps] for idx in s])       # (internal_batch, time)
        A_out = np.stack([A_out[idx : idx + n_time_steps, :] for idx in s])    # (internal_batch, time_solution, component)
        gradU = np.stack([gradU for i in s])                                   # (internal_batch, velocity_gradient)
    else:
        t_out = jnp.expand_dims(t_out, axis=0)
        gradU = np.expand_dims(gradU,  axis=0)
        A_out = jnp.expand_dims(A_out, axis=0)
    
    return t_out, gradU, A_out
    
# Transform data in dataloader to jax array
# Based on: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html
def jax_collate(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [jax_collate(samples) for samples in transposed]
    else:
        return jnp.asarray(batch)


#%% Mandel notation operations
@jax.jit
def tens2vec(t):
    """ 
        Transforms tensor into Mandel notation
        [A_{xx}, A_{yy}, A_{zz}, sqrt(2)*A_{yz}, sqrt(2)*A_{xz}, sqrt(2)*A_{xy}]
    """
    f = jnp.sqrt(2)
    return jnp.array([t[0,0], t[1,1], t[2,2], f*t[1,2], f*t[0,2], f*t[0,1]])

@jax.jit
def vec2tens(v):
    """ 
        Transforms a vector from Mandel notation to standard tensor notation
        A_{xx} A_{xy} A_{xz}
        A_{xy} A_{yy} A_{yz}
        A_{xz} A_{yz} A_{zz}
    """
    f = 1.0/jnp.sqrt(2)
    return jnp.array([[  v[0], f*v[5], f*v[4]],
                      [f*v[5],   v[1], f*v[3]],
                      [f*v[4], f*v[3],   v[2]]])

@jax.jit
def rotate4(A4_hat, Q):
    """
        Rotate A4_hat to laboratory frame
        A4 = Q_{im} Q_{jn} Q_{ko} Q_{lp} A4_hat_{mnop}
        
        based on: https://github.com/charlestucker3/Fiber-Orientation-Tools/blob/main/Fiber-Orientation-Tools/rotate4.m
        
        Literature: 
            Nadaud and Ferrari, 
            Invariant Tensor-to-Matrix Mappings for Evaluation of Tensorial Expressions,
            Journal of Elasticity 52: 43–61, 1998
        
    """
    mu = jnp.sqrt(2)
    f = 2.0/mu
    
    Q4a = jnp.array([[Q[0,0]*Q[0,0], Q[0,1]*Q[0,1], Q[0,2]*Q[0,2],  f*Q[0,1]*Q[0,2], f*Q[0,2]*Q[0,0], f*Q[0,0]*Q[0,1]],
                     [Q[1,0]*Q[1,0], Q[1,1]*Q[1,1], Q[1,2]*Q[1,2],  f*Q[1,1]*Q[1,2], f*Q[1,2]*Q[1,0], f*Q[1,0]*Q[1,1]],
                     [Q[2,0]*Q[2,0], Q[2,1]*Q[2,1], Q[2,2]*Q[2,2],  f*Q[2,1]*Q[2,2], f*Q[2,2]*Q[2,0], f*Q[2,0]*Q[2,1]],
                     
                     [mu*Q[1,0]*Q[2,0], mu*Q[1,1]*Q[2,1], mu*Q[1,2]*Q[2,2],  Q[2,2]*Q[1,1], Q[2,0]*Q[1,2], Q[2,1]*Q[1,0]],
                     [mu*Q[2,0]*Q[0,0], mu*Q[2,1]*Q[0,1], mu*Q[2,2]*Q[0,2],  Q[0,2]*Q[2,1], Q[0,0]*Q[2,2], Q[0,1]*Q[2,0]],
                     [mu*Q[0,0]*Q[1,0], mu*Q[0,1]*Q[1,1], mu*Q[0,2]*Q[1,2],  Q[1,2]*Q[0,1], Q[1,0]*Q[0,2], Q[1,1]*Q[0,0]]]) 

    
    Q4b = jnp.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, Q[1,2]*Q[2,1], Q[1,0]*Q[2,2], Q[1,1]*Q[2,0]],
                     [0, 0, 0, Q[2,2]*Q[0,1], Q[2,0]*Q[0,2], Q[2,1]*Q[0,0]],
                     [0, 0, 0, Q[0,2]*Q[1,1], Q[0,0]*Q[1,2], Q[0,1]*Q[1,0]]])
    
    Q4 = Q4a + Q4b
    
    
    A4rot = Q4 @ A4_hat @ Q4.T

    return A4rot

#%% Extract zip given path and folder_name
def extract_zipped_data(path_to_zip, intended_folder_name):
    folder_path = os.path.dirname(path_to_zip)
    
    path_to_data_folder = f"{folder_path}/{intended_folder_name}"
    
    exists_data_folder = os.path.exists(path_to_data_folder)
    exists_zipped_data = os.path.exists(path_to_zip)
    
    if not exists_data_folder and exists_zipped_data:
        os.mkdir(path_to_data_folder)
        
        with ZipFile(path_to_zip) as zObj:
            zObj.extractall(path = path_to_data_folder)
            
    if exists_data_folder and exists_zipped_data:
        if len(glob.glob(f"{path_to_data_folder}" + "/" + "*.h5")) == 0:
            with ZipFile(path_to_zip) as zObj:
                zObj.extractall(path = path_to_data_folder)
        else:
            pass
    
    if exists_data_folder and not exists_zipped_data:
        if glob.glob(f"{path_to_data_folder}" + "/" + "*.h5") != 0:
            pass
        else:
            raise RuntimeError("you have no DATA")
        
    return path_to_data_folder

#%% Main
if __name__ == "__main__":
    pass