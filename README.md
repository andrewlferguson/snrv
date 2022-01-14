snrv
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/snrv/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/snrv/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/snrv/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/snrv/branch/master)


State-free (non-)reversible VAMPnets

### Requirements

* numpy
* torch
* tqdm

### Installation

```
$ git clone https://github.com/andrewlferguson/snrv.git
$ pip install ./snrv
```
### Examples

Below is a quick start example that demonstrates basic usage of SNRVs.

Additional examples are provided in the Jupyter notebooks in this repo.

```python 
import numpy as np
from snrv import snrv 

etc
```

### Cite

If you use this code in your work, please cite:

W. Chen, H. Sidky, and A.L. Ferguson "Nonlinear discovery of slow molecular modes using state-free reversible VAMPnets" 
J. Chem. Phys. 150 214114 (2019) [doi: 10.1063/1.5092521](https://doi.org/10.1063/1.5092521)

```
@article{chen2019nonlinear,
  title={Nonlinear discovery of slow molecular modes using state-free reversible VAMPnets},
  author={Chen, Wei and Sidky, Hythem and Ferguson, Andrew L},
  journal={The Journal of chemical physics},
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

An older TensorFlow 1.15 version of SNRVs and without path reweighting is available [here](https://github.com/hsidky/srv).

**S(N)RVs:**

* W. Chen, H. Sidky, and A.L. Ferguson "Nonlinear discovery of slow molecular modes using state-free reversible 
VAMPnets" J. Chem. Phys. 150 214114 (2019) [doi: 10.1063/1.5092521](https://doi.org/10.1063/1.5092521)

**VAC / VAMP:**

* F. Noé "Machine learning for molecular dynamics on long timescales." Machine Learning Meets Quantum Physics. 
Springer, Cham, 2020. 331-372. [doi: 10.1007/978-3-030-40245-7_16](https://doi.org/10.1007/978-3-030-40245-7_16)

**VAMPnets / Deep CCA:**

* A. Mardt, L. Pasquali, H. Wu, and F. Noé "VAMPnets for deep learning of molecular kinetics" Nat. Commun. 9 5 (2018) 
[doi: 10.1038/s41467-017-02388-1](https://doi.org/10.1038/s41467-017-02388-1)

* Andrew, Galen, Raman Arora, Jeff Bilmes, and Karen Livescu. "Deep canonical correlation analysis." In International 
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

