import torch
from torch.utils.data import Dataset

__all__ = ["DatasetSnrv"]


class DatasetSnrv(Dataset):
    """
    Custom dataset for Snrv class

    Parameters
    ----------
    data : float tensor (single traj) or list of float tensors (multi traj); dim 0 = steps, dim 1 = features
        time-continuous trajectories

    lag : int
        lag in steps to apply to data trajectory

    ln_dynamical_weight : torch.tensor, n, n = observations
        accumulated sum of the log Girsanov path weights between frames in the trajectory;
        Girsanov theorem measure of the probability of the observed sample path under a target potential
        relative to that which was actually observed under the simulation potential;
        identically unity (no reweighting rqd) for target potential == simulation potential and code as None

    thermo_weight : torch.tensor, n, n = observations
        thermodynamic weights for each trajectory frame

    Attributes
    ----------
    self.lag : int
        lag in steps

    self.x_t0 : float tensor, n x dim, n = observations, dim = dimensionality of trajectory featurization
        non-time-lagged trajectory

    self.x_tt : float tensor, n x dim, n = observations, dim = dimensionality of trajectory featurization
        time-lagged trajectory

    self.pathweight : float tensor, n = observations
        pathweights from Girsanov theorem between time lagged observations;
        identically unity (no reweighting rqd) for target potential == simulation potential;
        if ln_pathweight == None => pathweight == ones
    """

    def __init__(self, data, lag, ln_dynamical_weight, thermo_weight):

        self.lag = lag

        if type(data) is list:

            for ii in range(0, len(data)):
                assert type(data[ii]) is torch.Tensor

            if (ln_dynamical_weight is not None) and (thermo_weight is not None):
                assert type(ln_dynamical_weight) is list
                assert type(thermo_weight) is list
                assert len(data) == len(ln_dynamical_weight) == len(thermo_weight)
                for ii in range(len(ln_dynamical_weight)):
                    assert type(ln_dynamical_weight[ii]) is torch.Tensor
                    assert type(thermo_weight[ii]) is torch.Tensor
                    assert (
                        data[ii].size()[0]
                        == ln_dynamical_weight[ii].size()[0]
                        == thermo_weight[ii].size()[0]
                    )

            x_t0 = list()
            x_tt = list()
            pathweight = list()

            for ii in range(len(data)):
                x_t0.append(data[ii][: -self.lag])
                x_tt.append(data[ii][self.lag :])

                K = data[ii][self.lag :].size()[0]
                pathweight_ii = torch.ones(K)
                if (ln_dynamical_weight is not None) and (thermo_weight is not None):
                    for jj in range(K):
                        arg = torch.sum(
                            ln_dynamical_weight[ii][jj + 1 : jj + self.lag + 1]
                        )
                        pathweight_ii[jj] = torch.exp(arg) * thermo_weight[ii][jj]
                pathweight.append(pathweight_ii)

            x_t0 = torch.cat(x_t0, dim=0)
            x_tt = torch.cat(x_tt, dim=0)
            pathweight = torch.cat(pathweight, dim=0)

        elif type(data) is torch.Tensor:

            if (ln_dynamical_weight is not None) and (thermo_weight is not None):
                assert type(ln_dynamical_weight) is torch.Tensor
                assert type(thermo_weight) is torch.Tensor
                assert (
                    data.size()[0]
                    == ln_dynamical_weight.size()[0]
                    == thermo_weight.size()[0]
                )

            x_t0 = data[: -self.lag]
            x_tt = data[self.lag :]

            K = x_tt.size()[0]
            pathweight = torch.ones(K)
            if (ln_dynamical_weight is not None) and (thermo_weight is not None):
                for ii in range(K):
                    arg = torch.sum(ln_dynamical_weight[ii + 1 : ii + self.lag + 1])
                    pathweight[ii] = torch.exp(arg) * thermo_weight[ii]

        else:
            raise TypeError(
                "Data type %s is not supported; must be a float tensor (single traj) or list of float tensors (multi "
                "traj)" % type(data)
            )

        self.x_t0 = x_t0
        self.x_tt = x_tt
        self.pathweight = pathweight

    def __getitem__(self, index):
        x_t0 = self.x_t0[index]
        x_tt = self.x_tt[index]
        pathweight = self.pathweight[index]
        return x_t0, x_tt, pathweight

    def __len__(self):
        assert len(self.x_t0) == len(self.x_tt) == len(self.pathweight)
        return len(self.x_t0)
