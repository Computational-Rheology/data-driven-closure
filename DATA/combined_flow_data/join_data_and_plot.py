# -*- coding: utf-8 -*-

#%%  mods
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#%% import

df_1 = pd.read_hdf("simple_shear_1e-05_0.01.h5", key="df")
df_2 = pd.read_hdf("shear_stretch_1e-05_0.01.h5", key="df")
df_3 = pd.read_hdf("shear_stretch2_1e-05_0.01.h5", key="df")

dfs = [df_1, df_2, df_3]


#%% Manipulate data
       
def append_data(col_name):
    tmp = df_1[col_name].to_numpy()
    tmp = np.append(tmp, df_2[col_name].iloc[1:].to_numpy())
    tmp = np.append(tmp, df_3[col_name].iloc[1:].to_numpy())
    return tmp

t = df_1["time"].to_numpy()
t = np.append(t, df_2["time"].iloc[1:].to_numpy() + t[-1])
t = np.append(t, df_3["time"].iloc[1:].to_numpy() + t[-1])

A11 = append_data("A_{11}")
A12 = append_data("A_{12}")
A13 = append_data("A_{13}")
A22 = append_data("A_{22}")
A23 = append_data("A_{23}")
A33 = append_data("A_{33}")


df_out = pd.DataFrame({"time": t, 
              "A_{11}": A11, "A_{12}": A12, "A_{13}": A13,
                             "A_{22}": A22, "A_{23}": A23,
                                            "A_{33}": A33})


df_out.to_hdf("combined_flow.h5", key="df")

plt.plot(t, A11)
plt.xlabel("time [s]")
plt.ylabel("$A_{11}$")
plt.savefig("combined_plot.png")
