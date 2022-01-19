snrv
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/andrewlferguson/snrv/workflows/CI/badge.svg)](https://github.com/andrewlferguson/snrv/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/snrv/badge/?version=latest)](https://snrv.readthedocs.io/en/latest/?badge=latest)
<!--- *Requires codecov.io account* [![codecov](https://codecov.io/gh/andrewlferguson/snrv/branch/master/graph/badge.svg)](https://codecov.io/gh/andrewlferguson/snrv/branch/master) --->


State-free (non-)reversible VAMPnets

### Requirements

* numpy
* scipy
* torch
* tqdm

### Environments

conda
```
$ conda env create --file requirements.yml
$ source activate snrv
```

venv
```
$ pip install virtualenv
$ python -m venv venv
$ source venv/bin/activate
$ python -m pip install -r requirements.txt
```

### Installation

```
$ git clone https://github.com/andrewlferguson/snrv.git
$ cd ./snrv
$ pip install .
OR
$ pip install -e .
```

### Examples

Below is a quick start minimal example.

Additional examples are provided in the Jupyter example notebooks.

```python 
import numpy as np
import torch
from snrv import Snrv, load_snrv

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# generating sythetic trajectory data: dim0 = observations, dim1 = features
dim = 3

# CASE 1: single trajectory
traj_x = np.random.randn(1000,dim)
traj_x = torch.from_numpy(traj_x).float()

# CASE 2: multiple trajectories packed into list
traj_x_1 = np.random.randn(1000,dim)
traj_x_1 = torch.from_numpy(traj_x_1).float()
traj_x_2 = np.random.randn(1000,dim)
traj_x_2 = torch.from_numpy(traj_x_2).float()
traj_x = [traj_x_1, traj_x_2]

# initializing S(N)RV model
input_size = dim
output_size = 2
n_epochs = 25
is_reversible = False

model = Snrv(input_size, output_size, n_epochs=n_epochs, is_reversible=is_reversible)
model = model.to(device)

# training model
lag = 1
model.fit(traj_x, lag)

# extracting implied time scales
its = -lag / np.log(model.evals.cpu().detach().numpy())

# projecting traj_x into transfer operator eigenvector approximations
psi = model.transform(traj_x)

# saving and loading trained model
model.save_model('model.pt')
model_two = load_snrv('model.pt')
```

### Cite

If you use this code in your work, please cite:

W. Chen, H. Sidky, and A.L. Ferguson "Nonlinear discovery of slow molecular modes using state-free reversible VAMPnets" 
J. Chem. Phys. 150 214114 (2019) [doi: 10.1063/1.5092521](https://doi.org/10.1063/1.5092521)

```
@article{chen2019nonlinear,
  title={Nonlinear discovery of slow molecular modes using state-free reversible VAMPnets},
  author={Chen, Wei and Sidky, Hythem and Ferguson, Andrew L},
  journal={The Journal of Chemical Physics},
  volume={150},
  number={21},
  pages={214114},
  year={2019},
  publisher={AIP Publishing LLC}
}
```


### Acknowledgements
 
The underlying mathematics of SNRVs are built upon the variational approach to conformational dynamics (VAC) and 
variational approach to Markov processes (VAMP) formalism and employ a form of deep canonical correlation analysis (CCA) 
upon time-lagged data. 

Numerically, they share similarities with [VAMPnets](https://github.com/markovmodel/deeptime) and 
[variational dynamics encoders (VDE)](https://github.com/msmbuilder/vde). Whereas VAMPnets seek to provide an end-to-end
replacement for learned state assigments and Markov state model (MSM) construction, SNRVs are designed to learn the slow 
modes within a dynamical trajectory. VDEs are limited to learning a single slow mode due to the lack of orthogonality 
constraints in the learned latedt space whereas SNRVs can scale to any dimensionality. 

SNRVs also incorporate functionality to 
perform path reweighting using the Girsanov formalism to estimate the slow modes under 
a target Hamiltonian from trajectories collected under a different, biased Hamiltonian. This situation is commonly
encountered when using enhanced sampling biasing techniques to enhance sampling of the phase space. Path weights
must be computed from the biased trajectories and provided as an input to the code.

An older TensorFlow 1.15 version of SNRVs without path reweighting is available [here](https://github.com/hsidky/srv).

**S(N)RVs:**

* W. Chen, H. Sidky, and A.L. Ferguson "Nonlinear discovery of slow molecular modes using state-free reversible 
VAMPnets" J. Chem. Phys. 150 214114 (2019) [doi: 10.1063/1.5092521](https://doi.org/10.1063/1.5092521)

**VAC / VAMP:**

* F. Noé "Machine learning for molecular dynamics on long timescales." Machine Learning Meets Quantum Physics. 
Springer, Cham, 2020. 331-372. [doi: 10.1007/978-3-030-40245-7_16](https://doi.org/10.1007/978-3-030-40245-7_16)

**VAMPnets / Deep CCA:**

* A. Mardt, L. Pasquali, H. Wu, and F. Noé "VAMPnets for deep learning of molecular kinetics" Nat. Commun. 9 5 (2018) 
[doi: 10.1038/s41467-017-02388-1](https://doi.org/10.1038/s41467-017-02388-1)

* G. Andrew, R. Arora, J. Bilmes, and K. Livescu. "Deep canonical correlation analysis." In International 
conference on machine learning, pp. 1247-1255. PMLR, (2013) [https://proceedings.mlr.press/v28/andrew13.html](https://proceedings.mlr.press/v28/andrew13.html)

**VDE:**

* C.X. Hernández, H.K. Wayment-Steele, M.M. Sultan, B.E. Husic, and V.S. Pande "Variational encoding of complex dynamics" 
Phys. Rev. E 97 6 062412 (2018) [doi: 10.1103/PhysRevE.97.062412](https://doi.org/10.1103/PhysRevE.97.062412)

**Girsanov path reweighting:**

* S. Kieninger and B.G. Keller "Path probability ratios for Langevin dynamics -- Exact and approximate" J. Chem. Phys. 
154 094102 (2021) [doi: 10.1063/5.0038408](https://doi.org/10.1063/5.0038408) 

* J.K. Weber and V.S. Pande "Potential-based dynamical reweighting for Markov state models of protein dynamics" J. Chem. Theory Comput. 11 6 2412–2420 (2015) [doi: 10.1021/acs.jctc.5b00031](https://doi.org/10.1021/acs.jctc.5b00031)

**Package structure:**

* Based on [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.


### License

The SNRV package is provided under a BSD-3-Clause license that can be found in the LICENSE file. By using, distributing, or 
contributing to this project, you agree to the terms and conditions of this license.

### Copyright

Copyright (c) 2022, Andrew Ferguson

