import numpy as np
import torch
from snrv import Snrv, load_snrv


def test_example_case1():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # generating sythetic trajectory data: dim0 = observations, dim1 = features
    dim = 3

    # CASE 1: single trajectory
    traj_x = np.random.randn(1000,dim)
    traj_x = torch.from_numpy(traj_x).float()

    # initializing S(N)RV model
    input_size = dim
    output_size = 2
    n_epochs = 3
    is_reversible = False

    model = Snrv(input_size, output_size, n_epochs=n_epochs, is_reversible=is_reversible, device=device)
    model = model.to(device)

    # training model
    lag = 1
    model.fit(traj_x, lag)

    # extracting implied time scales
    its = -lag / np.log(model.evals.cpu().detach().numpy())

    # projecting traj_x into transfer operator eigenvector approximations
    psi = [model.transform(x) for x in traj_x] if isinstance(traj_x, list) else model.transform(traj_x)

def test_example_case2():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # generating sythetic trajectory data: dim0 = observations, dim1 = features
    dim = 3

    # CASE 2: multiple trajectories packed into list
    traj_x_1 = np.random.randn(1000,dim)
    traj_x_1 = torch.from_numpy(traj_x_1).float()
    traj_x_2 = np.random.randn(1000,dim)
    traj_x_2 = torch.from_numpy(traj_x_2).float()
    traj_x = [traj_x_1, traj_x_2]

    # initializing S(N)RV model
    input_size = dim
    output_size = 2
    n_epochs = 3
    is_reversible = False

    model = Snrv(input_size, output_size, n_epochs=n_epochs, is_reversible=is_reversible, device=device)
    model = model.to(device)

    # training model
    lag = 1
    model.fit(traj_x, lag)

    # extracting implied time scales
    its = -lag / np.log(model.evals.cpu().detach().numpy())

    # projecting traj_x into transfer operator eigenvector approximations
    psi = [model.transform(x) for x in traj_x] if isinstance(traj_x, list) else model.transform(traj_x)