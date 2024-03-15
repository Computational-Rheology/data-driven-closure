## -*- coding: utf-8 -*-

#%% Modules
import jax
import jax.numpy as jnp

import diffrax
import equinox as eqx  

import optax  
    
import numpy as np
import torch

import ML_utilities as utils
from jax_custom_layer import Eigen_network

import matplotlib.pyplot as plt

from jax import config
config.update("jax_enable_x64", True)

import pandas as pd

#%% UDENetwork
class UDENetwork(eqx.Module):
    net : Eigen_network

    def __init__(self, net, layer_list, *, key, **kwargs):
        super().__init__(**kwargs)
        self.net = net(layer_list, key=key)
        
    def __call__(self, ts, y0, gradU):  
          
        solution = diffrax.diffeqsolve(
                                        diffrax.ODETerm(self.dy_dt),
                                        diffrax.Heun(),     # 2nd order scheme
                                        t0=ts[0],
                                        t1=ts[-1],
                                        dt0=None,   
                                        y0=y0,
                                        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5),
                                        saveat = diffrax.SaveAt(ts=ts),
                                        args = (gradU),
                                        throw=False     
                                    )
        
        # The solver will not throw an error.
        # If integration is successful, return solution. Else, vector of ones (empirical)
        results = jax.lax.select(diffrax.is_successful(solution.result), 
                                 solution.ys, 
                                 jnp.ones((len(ts), 6)))
                
        return results


    def dy_dt(self, t, Av, args):
        """
        Parameters
        ----------
        t : time
            scalar.
        Av : jax numpy array (6,)
            2nd-order orientation tensor in Mandel notation
        args : arguments to function
            

        Returns
        -------
        dA_dt in Mandel notation
        """
        # Comment for combined flow
        L = args
        L = L.reshape(3,3)     
        
        # # Uncomment for combined flow (post-process)
        # L_fnc = args
        # L = L_fnc(t)
        
        D = 0.5*(L + L.T)
        W = 0.5*(L - L.T)
                 
        A4, A = self.net(Av)
        
        dA_dt = ((W @ A) - (A @ W) 
                + 1.0*( 
                        (A @ D) + (D @ A)
                        -2.0*utils.vec2tens(A4 @ utils.tens2vec(D))
                    )) 
        
        return utils.tens2vec(dA_dt)
    
    def dy_dt_CGD(self, r, Av, args):
        # Post-processing for Center gated disk

        (U_func, L_fnc) = args
        L = L_fnc(r)
        U_r = U_func(r)
        
        D = 0.5*(L + L.T)
        W = 0.5*(L - L.T)
                 
        A4, A = self.net(Av)
        
        dA_dt = ((W @ A) - (A @ W) 
                + 1.0*( 
                        (A @ D) + (D @ A)
                        -2.0*utils.vec2tens(A4 @ utils.tens2vec(D))
                    )) 
        
        return utils.tens2vec(dA_dt)/U_r
        
#%% Seeds for consistency
seed = 1111
key = jax.random.PRNGKey(seed)
np.random.seed(seed)
torch.manual_seed(seed)

#%% Extract data for training and testing
z_train_data = 'DATA/train_data.zip'
z_test_data = 'DATA/test_data.zip'

train_data_path = utils.extract_zipped_data(z_train_data, "train_data")
test_data_path = utils.extract_zipped_data(z_test_data, "test_data")

#%% Create dataset, dataloader 

# Dataset for testing with UODE
# Will use increment of fits to gradually introduce more data into the system
dataset_params_dict ={"in_Mandel": True,  "skip_every":1, 
                      "n_time_steps":250, "n_initial_conditions":0}

dataset = utils.fiber_data(train_data_path, 
                     filter_fn = utils.filter_fn, 
                     key = key,
                     kwargs=(dataset_params_dict))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                         collate_fn=utils.jax_collate, 
                                         shuffle=True) 
    
print(f"DataLoader size: {len(dataloader)}")

#%% Create neuralNetwork
layers = [2, 10, 3]

model = UDENetwork(Eigen_network, layers, key=key)


for layer in model.net.layers:
    print(f'weights: {layer.weight}\n\n \
          bias: {layer.bias} \n ')

#%% Cost, step 
@eqx.filter_value_and_grad
def loss_func(model, ts, y0s, gradUs, ys_true):
    y_pred_inner_batch = jax.vmap(model, in_axes=(0, 0, 0))
    y_pred_outer_batch = jax.vmap(y_pred_inner_batch)
    ys_pred = y_pred_outer_batch(ts, y0s, gradUs)
    
    return jnp.sum((ys_true - ys_pred)**2) 


@eqx.filter_jit
def make_step(ts, y0s, gradUs, model, opt_state, ys_true, optim):
    loss, grads = loss_func(model, ts, y0s, gradUs, ys_true)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

loss_history = []
av_loss_history = []

#%% Training 1
epochs = 30

lr = 2e-3
optim = optax.adam(lr)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

for epoch in range(epochs):
    av_loss = 0
    for ts, gradU, ys_true in dataloader:
        y0s = ys_true[:, :, 0,:]              
        loss, model, opt_state = make_step(ts, y0s, gradU, 
                                            model, opt_state, ys_true, 
                                            optim)
        loss_history.append(loss)
        
        av_loss += loss
        
    av_loss_history.append(av_loss/len(dataloader))
    if epoch % 1  == 0:
        print(f'Epoch: {epoch}, loss: {av_loss/len(dataloader)}')
                    
#%% Dataset 
dataset_params_dict ={"in_Mandel": True,  "skip_every":2, 
                      "n_time_steps":250, "n_initial_conditions":0}

dataset = utils.fiber_data(train_data_path, 
                      filter_fn = utils.filter_fn, 
                      key = key,
                      kwargs=(dataset_params_dict))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                          collate_fn=utils.jax_collate, 
                                          shuffle=True) 
    
#%% Training 2
epochs = 30
lr = 1e-3

model2 = model
optim = optax.adam(lr)
opt_state = optim.init(eqx.filter(model2, eqx.is_inexact_array))

for epoch in range(epochs):
    av_loss = 0
    for ts, gradU, ys_true in dataloader:
        y0s = ys_true[:, :, 0,:]     
        loss, model2, opt_state = make_step(ts, y0s, gradU, 
                                            model2, opt_state, ys_true, 
                                            optim)
        
        loss_history.append(loss)
        
        av_loss += loss
    av_loss_history.append(av_loss/len(dataloader))
    if epoch % 1  == 0:
        print(f'Epoch: {epoch}, loss: {av_loss/len(dataloader)}')

#%% Dataset 
dataset_params_dict ={"in_Mandel": True,  "skip_every":3, 
                      "n_time_steps":250, "n_initial_conditions":0}

dataset = utils.fiber_data(train_data_path, 
                      filter_fn = utils.filter_fn, 
                      key = key,
                      kwargs=(dataset_params_dict))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                          collate_fn=utils.jax_collate, 
                                          shuffle=True) 
    
#%% Training 3
epochs = 20
lr = 5e-4

model3 = model2
optim = optax.adam(lr)
opt_state = optim.init(eqx.filter(model3, eqx.is_inexact_array))

for epoch in range(epochs):
    av_loss = 0
    for ts, gradU, ys_true in dataloader:
        y0s = ys_true[:, :, 0,:]
        
        loss, model3, opt_state = make_step(ts, y0s, gradU, 
                                            model3, opt_state, ys_true, 
                                            optim)
        
        loss_history.append(loss)
        
        av_loss += loss
    av_loss_history.append(av_loss/len(dataloader))
    if epoch % 1  == 0:
        print(f'Epoch: {epoch}, loss: {av_loss/len(dataloader)}')
        
#%% Dataset 
dataset_params_dict ={"in_Mandel": True,  "skip_every":4, 
                      "n_time_steps":250, "n_initial_conditions":0}

dataset = utils.fiber_data(train_data_path, 
                      filter_fn = utils.filter_fn, 
                      key = key,
                      kwargs=(dataset_params_dict))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                          collate_fn=utils.jax_collate, 
                                            shuffle=True) 
    
#%% Training 4
epochs = 10
lr = 1e-4

model4 = model3
optim = optax.adam(lr)
opt_state = optim.init(eqx.filter(model4, eqx.is_inexact_array))

for epoch in range(epochs):
    av_loss = 0
    for ts, gradU, ys_true in dataloader:
        y0s = ys_true[:, :, 0,:]
        
        loss, model4, opt_state = make_step(ts, y0s, gradU, 
                                            model4, opt_state, ys_true,
                                            optim)
        
        loss_history.append(loss)
        
        av_loss += loss
    av_loss_history.append(av_loss/len(dataloader))
    if epoch % 1  == 0:
        print(f'Epoch: {epoch}, loss: {av_loss/len(dataloader)}')         
     
        
#%% Save model
eqx.tree_serialise_leaves("model.eqx", model4)

#%% Pos-processing
# Load model
model = eqx.tree_deserialise_leaves("model.eqx", model) 

for layer in model.net.layers:
    print(f'weights: {layer.weight}\n\n \
          bias: {layer.bias} \n ')


#%% General plot settings
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fZ = 16
l=0.001
s=10
plt.rcParams['axes.linewidth'] = 1.75  
import matplotlib.patheffects as PathEffects

# https://matplotlib.org/stable/gallery/misc/patheffect_demo.html
line_eff = [PathEffects.Stroke(linewidth=2.5, foreground='k'), PathEffects.Normal()]
#%% Plot loss

# Plot average history data
plt.semilogy(av_loss_history, color="black")     
plt.yticks(fontsize = fZ)
plt.xticks(fontsize = fZ)
plt.xlabel("Iterations [-]", fontsize=fZ)
plt.ylabel("Av Loss", fontsize=fZ)

# Patches indicating the epochs
plt.vlines(29, 0, 0.5, color="red", linestyles="dashed")
plt.annotate(r"$1^{\mathrm{st}}$", (28, 0.6), fontsize=fZ-2)
plt.vlines(59, 0, 0.1, color="red", linestyles="dashed")
plt.annotate(r"$2^{\mathrm{nd}}$", (58, 0.11), fontsize=fZ-2)
plt.vlines(79, 0, 0.05, color="red", linestyles="dashed")
plt.annotate(r"$3^{\mathrm{rd}}$", (78, 0.06), fontsize=fZ-2)
plt.vlines(89, 0, 0.01, color="red", linestyles="dashed")
plt.annotate(r"$4^{\mathrm{th}}$", (87, 0.0125), fontsize=fZ-2)

plt.savefig("loss.svg", transparent=True)

#%% test data
path = "DATA/test_data"

test_dataset = utils.fiber_data(path, 
                                filter_fn = utils.filter_fn, 
                                key = key,
                                kwargs=({"in_Mandel":True,    "skip_every":None, 
                                         "n_time_steps":None, "n_initial_conditions":None}))

test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=1, 
                                              collate_fn=utils.jax_collate, 
                                              shuffle=False)  

print(f"DataLoader size: {len(test_dataloader)}")

for idx, (t, L, ys) in enumerate(test_dataloader):
    t = t.squeeze().squeeze()
    L = L.squeeze().squeeze()
    ys = ys.squeeze().squeeze()
    
    print(L)
    
    # Predict (solved in mandel notation)
    y_pred = model(t, ys[0,:], L)

    # Compute tensor and scalar error. Scalar error is:  sqrt(0.5 e_{ij} e_{ji}).
    err = ys - y_pred
    s_error  = np.array([jnp.sqrt(0.5*(tmp_err @ tmp_err)) for tmp_err in err])
    print(f" Av_error:  {np.round((1/t[-1])*np.trapz(s_error, x=t), 5)}")
    print(f" SS_error:  {np.round(s_error[-1], 5)}")
    
    # Back from mandel notation. Off diagonal entries need to divide by jnp.sqrt(2)
    f = 1.0/jnp.sqrt(2)
    y_pred = np.array(y_pred)
    ys = np.array(ys)
    
    y_pred[:,3] = y_pred[:,3]*f
    y_pred[:,4] = y_pred[:,4]*f
    y_pred[:,5] = y_pred[:,5]*f
    
    ys[:,3] = ys[:,3]*f
    ys[:,4] = ys[:,4]*f
    ys[:,5] = ys[:,5]*f
        
    plt.plot(t, y_pred[:, 0], label="$A_{11}^\mathrm{pred}$", color="red")
    plt.plot(t, y_pred[:, 1], label="$A_{22}^\mathrm{pred}$", color="blue")
    plt.plot(t, y_pred[:, 2], label="$A_{33}^\mathrm{pred}$", color="green")
    plt.plot(t, y_pred[:, 3], label="$A_{23}^\mathrm{pred}$", color="gray")
    plt.plot(t, y_pred[:, 4], label="$A_{13}^\mathrm{pred}$", color="darkviolet")
    plt.plot(t, y_pred[:, 5], label="$A_{12}^\mathrm{pred}$", color="orange")
    
    plt.plot(t, ys[:, 0], label="$A_{11}^\mathrm{true}$", color="red", linestyle="dashed", path_effects=line_eff)
    plt.plot(t, ys[:, 1], label="$A_{22}^\mathrm{true}$", color="blue", linestyle="dashed", path_effects=line_eff)
    plt.plot(t, ys[:, 2], label="$A_{33}^\mathrm{true}$", color="green", linestyle="dashed", path_effects=line_eff)
    plt.plot(t, ys[:, 3], label="$A_{23}^\mathrm{true}$", color="gray", linestyle="dashed", path_effects=line_eff)
    plt.plot(t, ys[:, 4], label="$A_{13}^\mathrm{true}$", color="darkviolet", linestyle="dashed", path_effects=line_eff)
    plt.plot(t, ys[:, 5], label="$A_{12}^\mathrm{true}$", color="orange", linestyle="dashed", path_effects=line_eff)
    
    plt.xlabel("Time [s]", fontsize=fZ+2)
    plt.ylabel("$A_{ij}$", fontsize=fZ+2)
    plt.xlim([0, 20])
    plt.ylim([0, 1.05])
    plt.xticks(fontsize=fZ)
    plt.yticks(fontsize=fZ)
    plt.legend(loc=4, fontsize=fZ, ncol=2, handleheight=1.75, labelspacing=0.05)
    plt.gca().tick_params(which = 'both', direction='out', length=5, width=1.75)
    plt.tight_layout()

    plt.savefig(f"{path}/{idx}.svg")
    plt.close()

    
#%% Test combined flow
# Restart kernel and update the function to solve 
combined_data_pdf = pd.read_hdf("DATA/combined_flow_data/combined_flow.h5", key="df")

def L(t):
        checks = jnp.array([t<10, t<20, t<=30])
        
        ss = jnp.array([[0, 1.0, 0],
                        [0, 0, 0],
                        [0, 0, 0]])
        
        s_str = jnp.array([[-1/20, 0, 0],
                            [0, -1/20, 1],
                            [0, 0, 1/10]])
        
        
        s_str2= np.array([[1, 0, 0],
                          [0, -1/2, 1],
                          [0, 0, -1/2]])
        
        results = jnp.array([ss, s_str, s_str2])
        
        index_selection = jnp.argmax(checks)
        return results[index_selection]
    
t = combined_data_pdf["time"].to_numpy()
# Sort in pseudo-Mandel notation
_ys = combined_data_pdf.filter(regex='A_{[1-6][1-6]}', axis=1).to_numpy()
ys = _ys[:, [0, 3, 5, 4, 2, 1]]

# initial condition in Mandel notation
y0 = ys[0,:] * np.array([1,1,1, np.sqrt(2), np.sqrt(2), np.sqrt(2)])

# Predict 
y_pred = model(t, y0, L)

# Back from Mandel notation
f = 1.0/jnp.sqrt(2)
y_pred = np.array(y_pred)

y_pred[:,3] = y_pred[:,3]*f
y_pred[:,4] = y_pred[:,4]*f
y_pred[:,5] = y_pred[:,5]*f

# Compute error
to_tensor = lambda x:  np.array([[x[0], x[5], x[4]],
                                 [x[5], x[1], x[3]],
                                 [x[4], x[3], x[2]]])


err = ys - y_pred
s_error  = np.array([np.sqrt(0.5*(np.tensordot(to_tensor(tmp_err), to_tensor(tmp_err)))) for tmp_err in err])
print(f"Av_error:  {np.round((1/t[-1])*np.trapz(s_error, x=t), 5)}")
print(f" SS_error:  {np.round(s_error[-1], 5)}")

plt.plot(t, y_pred[:,0], label="$A_{11}^\mathrm{pred}$", c="red", zorder=2)
plt.plot(t, y_pred[:,1], label="$A_{22}^\mathrm{pred}$", c="blue", zorder=2)

plt.plot(combined_data_pdf["time"], combined_data_pdf["A_{11}"], label="$A_{11}^\mathrm{true}$", color="red", linestyle="dashed", path_effects=line_eff)
plt.plot(combined_data_pdf["time"], combined_data_pdf["A_{22}"], label="$A_{22}^\mathrm{true}$", color="blue", linestyle="dashed", path_effects=line_eff)

plt.gca().tick_params(which = 'both', direction='out', length=5, width=1.75)
plt.xlabel("Time [s]", fontsize=fZ+2)
plt.ylabel("$A_{ij}$", fontsize=fZ+2)
plt.xlim([0,30])
plt.ylim([0,1.05])
plt.xticks(fontsize=fZ)
plt.yticks(fontsize=fZ)
plt.legend(loc=4, fontsize=fZ, ncol=2)
plt.tight_layout()
plt.savefig("combined_flow.svg")

#%% CGD
# Restart kernel and update the function to solve 
# Data for center gated disk is in "standard" notation: A_{11}, A_{12}, A_{13}, A_{22}, A_{23} and A_{33}
y0 = utils.tens2vec(jnp.eye(3)/3)
zs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Q = 1
b = 1

rs = np.arange(1, 20 + 0.005, 0.01)  

A_dict = {"A_{11}": 0,
          "A_{22}": 1,
          "A_{33}": 2,
          "A_{23}": 3,
          "A_{13}": 4,
          "A_{12}": 5}


to_tensor = lambda x:  np.array([[x[0], x[1], x[2]],
                                 [x[1], x[3], x[4]],
                                 [x[2], x[4], x[5]]])

for A_cmpt, c in [["A_{11}", "red"], ["A_{22}", "blue"]]:
    legend_counter = 0
    print(f"Solving for component: {A_cmpt}")
    for z in zs:
        df = pd.read_hdf(f"DATA/CGD/{z}_CGD.h5", key="df")
        def U_CGD(r, z=z, Q=Q, b=b):
            tmp = (3*Q)/(8*jnp.pi*r*b) 
            return tmp*(1-(z*z)/(b*b))
        
        def L(r, z=z, Q=Q, b=b):
              tmp = (3*Q)/(8*jnp.pi*r*b)
              z_b_square = (z*z)/(b*b)
              xx = (1/r)*(1 - z_b_square)
             
              arr = tmp*jnp.array([[-xx, 0, -(2/b)*(z/b)],
                                   [0, xx, 0],
                                   [0, 0, 0]])
             
              return arr
         
        y_pred = model(rs, y0, (U_CGD,L))    
        
        # Compute error and scalar error magnitude        
        ys = df.filter(regex='A_{[1-6][1-6]}', axis=1).to_numpy()
        ys = np.array([to_tensor(tmp) for tmp in ys])
        y_pred_tmp = np.array([utils.vec2tens(tmp) for tmp in y_pred])
        err = ys - y_pred_tmp
        s_error  = np.array([np.sqrt(0.5*(np.tensordot(tmp_err, tmp_err))) for tmp_err in err])
        
        if A_cmpt == "A_{11}":
            print(f"z = {z}. Error is: {np.round((1/(rs[-1] - rs[0]))*np.trapz(s_error, x=rs), 5)}")
            print(f" SS_error:  {np.round(s_error[-1], 5)}")
            print(f" A_11_diff: {np.round(ys[-1,0,0] - y_pred[-1,0], 5)}")
        
        # Back from Mandel notation
        f = 1.0/jnp.sqrt(2)
        y_pred = np.array(y_pred)

        y_pred[:,3] = y_pred[:,3]*f
        y_pred[:,4] = y_pred[:,4]*f
        y_pred[:,5] = y_pred[:,5]*f
         
        A_idx = A_dict[A_cmpt]
        plt.plot(rs, y_pred[:, A_idx], label=f"${A_cmpt}" + "^\mathrm{pred}$" if legend_counter == 0 else "_nolegend_",
                  color=c)
        plt.plot(df["radius"], df[A_cmpt], label=f"${A_cmpt}" +"^\mathrm{true}$" if legend_counter == 0 else "_nolegend_", linestyle="dashed",
                  color=c, path_effects=line_eff)
        
        r_point = 15
        closest_y_values_to_r_point = (df["radius"] - r_point).abs().argmin()
        A_11_closest_value = df.iloc[closest_y_values_to_r_point][A_cmpt]
        plt.annotate(r"$\bar{x}_3 =$" + f"{z}", (r_point, 0.99*A_11_closest_value), fontsize=fZ-2,
                      bbox=dict(boxstyle='square, pad=.5', 
                                facecolor="white", 
                                edgecolor="white"))
        
        legend_counter+=1
        
    plt.gca().tick_params(which = 'both', direction='out', length=5, width=1.75)
    plt.xlim([0, 20])
    if A_cmpt == "A_{13}":
        plt.ylim([-0.5, 0])
    else:
        plt.ylim([0, 1])    
    plt.xlabel(r"$\bar{x}_1$", fontsize=fZ+2)
    plt.ylabel(f"${A_cmpt}$", fontsize=fZ+2)
    plt.xticks(fontsize=fZ)
    plt.yticks(fontsize=fZ)
    plt.legend(ncols=2 , fontsize=fZ, loc=2)
    plt.tight_layout()
    plt.savefig(f"{A_cmpt}_CGD.svg")
    plt.close() 
    
