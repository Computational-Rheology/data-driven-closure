This repository finds a closure for the 4th-order tensor used in Jeffery's equation.

It assumes that 4th-order tensor is orthotropic. 

Admissible ranges are derived based on Mathematica&#174; 14.


----

Folders:
* **DATA** &rarr; Contains .h5 files for training, testing and post-processing
* **Admissibility** &rarr; Contains a python script with functions in Mandel notation and Sympy derivation of the bounds

----

Files:
* *ML_utilities.py* &rarr; Python script with utilities (dataset, functions) for machine learning
  
* *jax_custom_layer.py* &rarr; Python script with "eigenlayer" to limit the search space of $d_1$, $d_2$, $d_3$

* *model_UODE.py* &rarr; Python script with UODE, training procedure and post-processing


----
Packages:

*   Diffrax: '0.5.0'
*   JAX: '0.4.23'
*   Equinox: '0.11.3'
*   Optax: '0.1.9'
*   Pytorch: '2.2.0'
*   Numpy: '1.24.3'



