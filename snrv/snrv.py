if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("0")

import numpy as np
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

from tqdm import tqdm

from snrv.data import DatasetSnrv
from snrv.utils import (
    accumulate_correlation_matrices,
    gen_eig_chol,
    stable_symmetric_inverse,
)

__all__ = ["Snrv", "load_snrv"]


class Snrv(nn.Module):
    """
    ANN encoder for state-free reversible VAMPnet

    Parameters
    ----------
    input_size : int
        number of neurons in input layer

    output_size : int
        number of neurons in output layer corresponding to number of basis functions to compute for VAC / VAMP

    hidden_depth : int, default = 2
        number of hidden layers

    hidden_size : int, default = 100
        number of neurons in each hidden layer

    activation : str, default = nn.ReLU()
        activation function to be used in each neuron

    batch_norm : bool, default = False
        use batch normalization after each layer

    dropout_rate : float, default = 0.
        dropout rate to use in each layer

    lr : float, default = 0.1
        learning rate for Adam optimizer

    weight_decay : float, default = 0.
        weight decay for Adam optimizer

    val_frac : float, default = 0.2
        fraction of data to place in validation partition; balance is used for training

    n_epochs : int, default = 100
        number of training epochs for ANN

    batch_size : int, default = 100
        no. of instances per mini batch

    VAMPdegree: int, default = 2
        exponent to use in VAMP-r score loss function

    is_reversible : bool, default = True
        indicator flag as to whether to enforce detailed balance by symmetrizing trajectory by augmenting with time
        reversed twin and solve VAC OR not assume microscopic reversibility and solve VAMP

    num_workers : int, default = 8
        number of cpu workers to use for datalaoder

    Attributes
    ----------
    self.device : str
        specification as to whether to use 'cpu' or 'cuda' for model training

    self.is_fitted : bool
        indicator flag as to whether or nor model has been fitted

    self._train_loader : DataLoader object
        training data loader

    self._val_loader : DataLoader object
        validation data loader

    self.lag : int
        lag in steps to apply to data trajectory

    self.optimizer : torch.optim.Adam object
        optimizer for backpropagation

    self._train_step : function
        training step function

    self.evals : torch.tensor, n_comp, n_comp = no. of basis functions in ANN == output_size
        eigenvalues of VAC generalized eigenvalue problem finding linear combination of learned basis functions to
        produce approximations of transfer operator eigenvectors in non-ascending order OR singular values of VAMP
        singular value problem finding linear combination of learned basis functions to produce approximations of
        transfer operator left and right singular vectors in non-ascending order

    self.expansion_coefficients : n_comp, n_comp = no. of basis functions in ANN == output_size
        expansion coefficients for linear combination of learned basis functions into transfer operator eigenvectors
        (reversible) or left singular vectors (non-reversible)

    self.expansion_coefficients_right : n_comp, n_comp = no. of basis functions in ANN == output_size
        expansion coefficients for linear combination of learned basis functions into transfer operator right
        singular vectors (non-reversible)

    self.training_losses : list, n_epoch
        loss over training data in each epoch

    self.validation_losses : list, n_epoch
        loss over validation data in each epoch
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_depth=2,
        hidden_size=100,
        activation=nn.ReLU(),
        batch_norm=False,
        dropout_rate=0.0,
        lr=0.1,
        weight_decay=0.0,
        val_frac=0.2,
        n_epochs=100,
        batch_size=100,
        VAMPdegree=2,
        is_reversible=True,
        num_workers=8,
    ):

        super().__init__()

        assert 0.0 < val_frac < 1.0
        assert isinstance(VAMPdegree, int) and VAMPdegree > 0

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_depth = hidden_depth
        self.hidden_size = hidden_size
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_frac = val_frac
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.VAMPdegree = VAMPdegree
        self.is_reversible = is_reversible

        # optimize number of workers for data loading
        if num_workers is None:
            self.num_workers = min(os.cpu_count(), 8)
        else:
            self.num_workers = num_workers

        # building SNRV encoder as simple feedforward ANN
        self.model = list()
        self.model.append(nn.Linear(self.input_size, self.hidden_size))
        if self.dropout_rate > 0.0:
            self.model.append(nn.Dropout(self.dropout_rate))
        if self.batch_norm:
            self.model.append(nn.BatchNorm1d(self.hidden_size))
        self.model.append(self.activation)
        for k in range(hidden_depth - 1):
            self.model.append(nn.Linear(self.hidden_size, self.hidden_size))
            if self.dropout_rate > 0.0:
                self.model.append(nn.Dropout(self.dropout_rate))
            if self.batch_norm:
                self.model.append(nn.BatchNorm1d(self.hidden_size))
            self.model.append(self.activation)
        self.model.append(nn.Linear(self.hidden_size, self.output_size))
        self.model = nn.Sequential(*self.model)

        # cached variables
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_fitted = False
        self._train_loader = None
        self._val_loader = None
        self.lag = None
        self.optimizer = None
        self.training_losses = None
        self.validation_losses = None
        self.evals = None
        self.expansion_coefficients = None
        self.expansion_coefficients_right = None

    def forward(self, x_t0, x_tt):
        """
        Forward pass through network

        Parameters
        ----------
        x_t0 : torch.tensor, n x dim, n = observations, dim = dimensionality of trajectory featurization
            trajectory

        x_tt : torch.tensor, n x dim, n = observations, dim = dimensionality of trajectory featurization
            time-lagged trajectory of same length as trajectory

        Return
        ------
        z_t0 : torch.tensor, n x n_comp, n = observations, n_comp = number of basis functions produced by network
            trajectory projected into basis functions learned by SNRV encoder

        z_tt : torch.tensor, n x n_comp, n = observations, n_comp = number of basis functions produced by network
            time-lagged trajectory of same length as trajectory projected into basis functions learned by SNRV encoder
        """
        assert x_tt.size()[0] == x_t0.size()[0]

        # passing inputs through ANN to project into self.output_size basis vectors
        x_t0 = self.model(x_t0)
        x_tt = self.model(x_tt)

        return (
            x_t0,
            x_tt,
        )  # really z_t0, z_tt resulting from passing x_t0, x_tt through network

    def _loss_fn(self, z_t0, z_tt, pathweight):
        """
        VAC / VAMP VAMP-r loss function

        Parameters
        ----------
        z_t0 : torch.tensor, n x n_comp, n = observations, n_comp = number of basis functions produced by network
            trajectory projected into basis functions learned by SNRV encoder

        z_tt : torch.tensor, n x n_comp, n = observations, n_comp = number of basis functions produced by network
            time-lagged trajectory of same length as trajectory projected into basis functions learned by SNRV encoder

        pathweight : float tensor, n = observations
            pathweights from Girsanov theorem between time lagged observations;
            identically unity (no reweighting rqd) for target potential == simulation potential

        Return
        ------
        loss : float
            negative squared sum of eigenvalues from solving VAC generalized eigenvalue
            problem OR VAMP singular value problem
        """

        assert z_t0.size()[0] == z_tt.size()[0] == pathweight.size()[0]
        assert z_t0.size()[1] == z_tt.size()[1]

        # VAC / VAMP on instantaneous z_t0 and time-lagged z_tt trajectories projected into ANN basis functions
        # Ref.: Noe arXiv:1812.07669v1
        dim = z_t0.size()[1]

        # - accumulating correlation matrices
        C00 = torch.zeros(dim, dim, device=self.device)
        C01 = torch.zeros(dim, dim, device=self.device)
        C10 = torch.zeros(dim, dim, device=self.device)
        C11 = torch.zeros(dim, dim, device=self.device)
        C00, C01, C10, C11 = accumulate_correlation_matrices(
            z_t0, z_tt, pathweight, C00, C01, C10, C11
        )

        if self.is_reversible:

            # VAC
            # Ref.: Noe arXiv:1812.07669v1

            # - assuming detailed balance (i.e., data augmentation with time reversed trajectories)
            Q = 0.5 * (C00 + C11)
            C = 0.5 * (C01 + C10)

            # - applying regularization (nugget regularization of unpopulated bins with pseudocounts)
            # Q += torch.eye(Q.size()[0], dtype=torch.float, requires_grad=False)*torch.finfo(torch.float32).eps

            # solving generalized eigenvalue problem Cv = wQv using Cholesky trick to enable backpropagation
            evals, _ = gen_eig_chol(C, Q)

            # loss
            loss = -(evals ** self.VAMPdegree).sum()

        else:

            # VAMP
            # Ref.: Noe arXiv:1812.07669v1, Algorithm 4

            # - assembling balanced propagator (Eqn. 34)
            C00invhalf = stable_symmetric_inverse(C00, ret_sqrt=True)
            C11invhalf = stable_symmetric_inverse(C11, ret_sqrt=True)

            P = torch.matmul(C00invhalf, torch.matmul(C01, C11invhalf))

            # - SVD
            Up, S, VpT = torch.linalg.svd(P)

            # - projecting singular values back to original (non-balanced) propagator
            # U = torch.matmul(C00invhalf, Up)
            # V = torch.matmul(C11invhalf, VpT.t())

            # loss
            loss = -(S ** self.VAMPdegree).sum()

        return loss

    def _create_dataset(self, data, ln_dynamical_weight, thermo_weight):
        """
        create training and validation data loader

        Parameters
        ----------
        data : torch.tensor, n x dim, n = observations, dim = dimensionality of trajectory featurization
            trajectory

        ln_pathweight : torch.tensor, n, n = observations
            accumulated sum of the log Girsanov path weights between frames in the trajectory;
                Girsanov theorem measure of the probability of the observed sample path under a target potential
                relative to that which was actually observed under the simulation potential;
                identically unity (no reweighting rqd) for target potential == simulation potential and code as None;
            Ref.: Kieninger and Keller J. Chem. Phys 154 094102 (2021)  https://doi.org/10.1063/5.0038408

        Return
        ------
        self._train_loader : torch DataLoader
            training data loader

        self._val_loader : torch DataLoader
            validation data loader
        """
        dataset = DatasetSnrv(data, self.lag, ln_dynamical_weight, thermo_weight)

        n = len(dataset)
        train_size = int((1.0 - self.val_frac) * n)
        val_size = n - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self._train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self._val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return None

    def _train_step(self, x_t0, x_tt, pathweight):
        """
        model training step

        Parameters
        ----------
        x_t0: torch.tensor, n x dim, n = observations, dim = dimensionality of trajectory featurization
            trajectory

        x_tt: torch.tensor, n x dim, n = observations, dim = dimensionality of trajectory featurization
            time-lagged trajectory of same length as trajectory

        pathweight : float tensor, n = observations
            pathweights from Girsanov theorem between time lagged observations;
            identically unity (no reweighting rqd) for target potential == simulation potential

        Return
        ------
        loss.item() : float
            network loss over x_t0 and x_tt mini batch extracted as a float
        """
        z_t0, z_tt = self(x_t0, x_tt)
        loss = self._loss_fn(z_t0, z_tt, pathweight)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def fit(self, data, lag, ln_dynamical_weight=None, thermo_weight=None):
        """
        fit SNRV model to data

        Parameters
        ----------
        data : torch.tensor, n x dim, n = observations, dim = dimensionality of trajectory featurization
            trajectory

        lag : int
            lag in steps to apply to data trajectory

        ln_dynamical_weight : torch.tensor, n, n = observations, default = None
            accumulated sum of the log Girsanov path weights between frames in the trajectory;
            Girsanov theorem measure of the probability of the observed sample path under a target potential
            relative to that which was actually observed under the simulation potential;
            identically unity (no reweighting rqd) for target potential == simulation potential and code as None;
            Ref.: Kieninger and Keller J. Chem. Phys 154 094102 (2021)  https://doi.org/10.1063/5.0038408

        thermo_weight : torch.tensor, n, n = observations
            thermodynamic weights for each trajectory frame

        Return
        ------
        self.lag : int
            lag in steps

        self.training_losses: list, n_epoch
            loss over training data in each epoch

        self.validation_losses: list, n_epoch
            loss over validation data in each epoch

        self.evals : torch.tensor, n_comp, n_comp = no. of basis functions in ANN == output_size
            eigenvalues of VAC generalized eigenvalue problem finding linear combination of learned basis functions to
            produce approximations of transfer operator eigenvectors in non-ascending order OR singular values of VAMP
            singular value problem finding linear combination of learned basis functions to produce approximations of
            transfer operator left and right singular vectors in non-ascending order

        self.expansion_coefficients : n_comp, n_comp = no. of basis functions in ANN == output_size
            expansion coefficients for linear combination of learned basis functions into transfer operator eigenvectors
            (reversible) or left singular vectors (non-reversible)

        self.expansion_coefficients_right : n_comp, n_comp = no. of basis functions in ANN == output_size
            expansion coefficients for linear combination of learned basis functions into transfer operator right
            singular vectors (non-reversible)

        self.is_fitted: bool
            indicator flag as to whether or nor model has been fitted
        """

        assert isinstance(lag, int) and lag >= 1

        self.lag = lag
        self._create_dataset(data, ln_dynamical_weight, thermo_weight)
        self.optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        training_losses = list()
        validation_losses = list()

        for epoch in range(self.n_epochs):

            with tqdm(self._train_loader, unit="batch") as tepoch:

                self.train()
                train_losses = list()
                for x_t0_batch, x_tt_batch, pathweight_batch in tepoch:
                    tepoch.set_description("Epoch %d" % epoch)
                    x_t0_batch = x_t0_batch.to(self.device)
                    x_tt_batch = x_tt_batch.to(self.device)
                    pathweight_batch = pathweight_batch.to(self.device)
                    loss = self._train_step(x_t0_batch, x_tt_batch, pathweight_batch)
                    train_losses.append(loss)
                training_loss = float(np.mean(train_losses))
                training_losses.append(training_loss)

                self.eval()
                with torch.no_grad():
                    val_losses = []
                    for x_t0_batch, x_tt_batch, pathweight_batch in self._val_loader:
                        x_t0_batch = x_t0_batch.to(self.device)
                        x_tt_batch = x_tt_batch.to(self.device)
                        pathweight_batch = pathweight_batch.to(self.device)
                        z_t0_batch, z_tt_batch = self(x_t0_batch, x_tt_batch)
                        val_loss = self._loss_fn(
                            z_t0_batch, z_tt_batch, pathweight_batch
                        )
                        val_loss = val_loss.item()
                        val_losses.append(val_loss)
                    validation_loss = float(np.mean(val_losses))
                    validation_losses.append(validation_loss)

                print(
                    "[Epoch %d]\t training loss = %.3f\t validation loss = %.3f"
                    % (epoch, training_loss, validation_loss)
                )

        self.training_losses = training_losses
        self.validation_losses = validation_losses

        self._compute_expansion_coefficients()

        self.is_fitted = True

        return None

    def _compute_expansion_coefficients(self):
        """
        calculate expansion coefficients by applying trained SNRV encoder ANN to all training data

        Parameters
        ----------

        Return
        ------
        self.evals : torch.tensor, n_comp, n_comp = no. of basis functions in ANN == output_size
            eigenvalues of VAC generalized eigenvalue problem finding linear combination of learned basis functions to
            produce approximations of transfer operator eigenvectors in non-ascending order OR singular values of VAMP
            singular value problem finding linear combination of learned basis functions to produce approximations of
            transfer operator left and right singular vectors in non-ascending order

        self.expansion_coefficients : n_comp, n_comp = no. of basis functions in ANN == output_size
            expansion coefficients for linear combination of learned basis functions into transfer operator eigenvectors
            (reversible) or left singular vectors (non-reversible)

        self.expansion_coefficients_right : n_comp, n_comp = no. of basis functions in ANN == output_size
            expansion coefficients for linear combination of learned basis functions into transfer operator right
            singular vectors (non-reversible)
        """

        C00 = torch.zeros(self.output_size, self.output_size, device=self.device)
        C01 = torch.zeros(self.output_size, self.output_size, device=self.device)
        C10 = torch.zeros(self.output_size, self.output_size, device=self.device)
        C11 = torch.zeros(self.output_size, self.output_size, device=self.device)

        self.eval()
        with torch.no_grad():
            for x_t0_batch, x_tt_batch, pathweight_batch in self._train_loader:
                x_t0_batch = x_t0_batch.to(self.device)
                x_tt_batch = x_tt_batch.to(self.device)
                pathweight_batch = pathweight_batch.to(self.device)
                z_t0_batch, z_tt_batch = self(x_t0_batch, x_tt_batch)
                C00, C01, C10, C11 = accumulate_correlation_matrices(
                    z_t0_batch, z_tt_batch, pathweight_batch, C00, C01, C10, C11
                )

        if self.is_reversible:

            # VAC
            # Ref.: Noe arXiv:1812.07669v1

            # assuming detailed balance (i.e., data augmentation with time reversed trajectories)
            Q = 0.5 * (C00 + C11)
            C = 0.5 * (C01 + C10)

            # applying regularization (nugget regularization of unpopulated bins with pseudocounts)
            # Q += torch.eye(Q.size()[0], dtype=torch.float, requires_grad=False)*torch.finfo(torch.float32).eps

            # solving generalized eigenvalue problem Cv = wQv using Cholesky trick to enable backpropagation
            # - column evecs are the expansion coefficients to assemble transfer operator eigenvector / singluar vector
            # approximations from learned SNRV basis functions
            evals, expansion_coefficients = gen_eig_chol(C, Q)

            self.evals = evals
            self.expansion_coefficients = expansion_coefficients

        else:

            # VAMP
            # Ref.: Noe arXiv:1812.07669v1, Algorithm 4

            # - assembling balanced propagator (Eqn. 34)
            C00invhalf = stable_symmetric_inverse(C00, ret_sqrt=True)
            C11invhalf = stable_symmetric_inverse(C11, ret_sqrt=True)

            P = torch.matmul(C00invhalf, torch.matmul(C01, C11invhalf))

            # - SVD
            Up, S, VpT = torch.linalg.svd(P)

            # - projecting singular values back to original (non-balanced) propagator
            U = torch.matmul(C00invhalf, Up)
            V = torch.matmul(C11invhalf, VpT.t())

            self.evals = S
            self.expansion_coefficients = U
            self.expansion_coefficients_right = V

        return None

    def transform(self, data):
        """
        project data into learned eigenvector basis

        Parameters
        ----------
        data : torch.tensor, n x dim, n = observations, dim = dimensionality of trajectory featurization
            trajectory

        Return
        ------
        psi : torch.tensor, n x n_comp, n_comp = no. of basis functions in ANN == output_size
            projection of data into learned eigenvector approximations of transfer operator
        """

        if self.is_fitted:
            #with torch.no_grad(): # need grad for get_transform_Jacobian to function
            data = data.to(self.device)
            z, _ = self(data, data)
            psi = torch.matmul(z, self.expansion_coefficients)
        else:
            raise RuntimeError("Model needs to be fit first.")

        return psi

    def fit_transform(self, data, lag, ln_dynamical_weight=None, thermo_weight=None):
        """
        fit SNRV over data then project data into learned eigenvector (VAC) / singular vectors (VAMP)
        of transfer operator

        Parameters
        ----------
        data : torch.tensor, n x dim, n = observations, dim = dimensionality of trajectory featurization
            trajectory

        lag : int
            lag in steps to apply to data trajectory

        ln_dynamical_weight : torch.tensor, n, n = observations, default = None
            accumulated sum of the log Girsanov path weights between frames in the trajectory;
            Girsanov theorem measure of the probability of the observed sample path under a target potential
            relative to that which was actually observed under the simulation potential;
            identically unity (no reweighting rqd) for target potential == simulation potential and code as None;
            Ref.: Kieninger and Keller J. Chem. Phys 154 094102 (2021)  https://doi.org/10.1063/5.0038408

        thermo_weight : torch.tensor, n, n = observations
            thermodynamic weights for each trajectory frame

        Return
        ------
        self.lag : int
            lag in steps

        self.training_losses: list, n_epoch
            loss over training data in each epoch

        self.validation_losses: list, n_epoch
            loss over validation data in each epoch

        self.evals : torch.tensor, n_comp, n_comp = no. of basis functions in ANN == output_size
            eigenvalues of VAC generalized eigenvalue problem finding linear combination of learned basis functions to
            produce approximations of transfer operator eigenvectors in non-ascending order OR singular values of VAMP
            singular value problem finding linear combination of learned basis functions to produce approximations of
            transfer operator left and right singular vectors in non-ascending order

        self.expansion_coefficients : n_comp, n_comp = no. of basis functions in ANN == output_size
            expansion coefficients for linear combination of learned basis functions into transfer operator eigenvectors
            (reversible) or left singular vectors (non-reversible)

        self.expansion_coefficients_right : n_comp, n_comp = no. of basis functions in ANN == output_size
            expansion coefficients for linear combination of learned basis functions into transfer operator right
            singular vectors (non-reversible)

        self.is_fitted: bool
            indicator flag as to whether or nor model has been fitted

        psi : torch.tensor, n x n_comp, n_comp = no. of basis functions in ANN == output_size
            projection of data into learned eigenvector approximations of transfer operator
        """

        assert isinstance(lag, int) and lag >= 1

        self.lag = lag
        self.fit(
            data,
            self.lag,
            ln_dynamical_weight=ln_dynamical_weight,
            thermo_weight=thermo_weight,
        )
        psi = self.transform(data)

        return psi

    def get_transform_Jacobian(self, data):
        """
        compute Jacobian of self.transform computational graph output (psi) wrt input (data)

        - data is n x dim_in tensor -- n instances of dim_in = self.input_size vectors at which to compute Jacobian
        - psi = self.transform(data) is n x dim_out -- n dim_out = n_comp = no. basis functions in ANN
          = self.output_size projections of data vectors
        - Jacobian[n,i,j] = d(psi_i)/d(data_j) @ data[n,:]

        Parameters
        ----------
        data : torch.tensor, n x dim_in, n = observations, dim_in = dimensionality of trajectory
            featurization = self.input_size

        Return
        ------
        Jacobian : torch.tensor, n x dim_out x dim_in, n = observations, dim_in = dimensionality of input,
            dim_out = dimensionality of output
        """

        if not self.is_fitted:

            raise RuntimeError("Model needs to be fit first.")

        else:

            # passing data through self.transform computational graph with gradients
            data.requires_grad = True
            psi = self.transform(data)

            # preparing Jacobian
            n = data.size()[0]
            dim_out = self.output_size
            dim_in = data.size()[1]

            Jacobian = torch.zeros(n, dim_out, dim_in)

            # sequential computation of Jacobian[:,ii,:] (i.e., row ii of each Jacobian corresponding to each n)
            for ii in range(dim_out):

                # - selecting the computational graph output for which to compute gradient wrt all inputs
                #   (i.e., row of 2D dim_out x dim_in Jacobian for each ii = 0...n)
                grad_mask = torch.zeros(dim_out).to(self.device)
                grad_mask[ii] = 1
                grad_mask_matrix = torch.tile(grad_mask, (n, 1)).to(self.device)

                # - zeroing gradients
                if hasattr(data.grad, "data"):
                    _ = data.grad.data.zero_()

                # - backward pass
                psi.backward(grad_mask_matrix, retain_graph=True)

                # - computing gradients and storing in Jacobian
                Jacobian[:, ii, :] = torch.clone(data.grad.data)

        return Jacobian

    def save_model(self, modelFilePath):
        """
        saving model parameters required for building and running (i.e., self.transform)

        Parameters
        ----------
        modelFilePath : str
            path to .pt file

        Return
        ------
        """
        torch.save(
            {
                "input_size": self.input_size,
                "output_size": self.output_size,
                "hidden_depth": self.hidden_depth,
                "hidden_size": self.hidden_size,
                "activation": self.activation,
                "batch_norm": self.batch_norm,
                "dropout_rate": self.dropout_rate,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "val_frac": self.val_frac,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "VAMPdegree": self.VAMPdegree,
                "is_reversible": self.is_reversible,
                "evals": self.evals,
                "expansion_coefficients": self.expansion_coefficients,
                "network_weights": self.state_dict(),
            },
            modelFilePath,
        )

        return None

    def load_weights(self, network_weights):
        """
        loading SNRV network weights from file

        Parameters
        ----------
        network_weights : ordered dictionary
            dictionary of network_weights
            (N.B. model must possess same architecture as that used to save these weights!)

        Return
        ------
        """
        self.load_state_dict(network_weights)
        self.is_fitted = True
        self.eval()

        return None


def load_snrv(modelFilePath):
    """
    loading SNRV model from file

    Parameters
    ----------
    modelFilePath : str
        path to .pt file containing saved model parameters

    Return
    ------
    model : Snrv object
        SNRV model initialized from saved parameter set
    """
    d = torch.load(modelFilePath)

    input_size = d["input_size"]
    output_size = d["output_size"]
    hidden_depth = d["hidden_depth"]
    hidden_size = d["hidden_size"]
    activation = d["activation"]
    batch_norm = d["batch_norm"]
    dropout_rate = d["dropout_rate"]
    lr = d["lr"]
    weight_decay = d["weight_decay"]
    val_frac = d["val_frac"]
    n_epochs = d["n_epochs"]
    batch_size = d["batch_size"]
    VAMPdegree = d["VAMPdegree"]
    is_reversible = d["is_reversible"]

    evals = d["evals"]
    expansion_coefficients = d["expansion_coefficients"]
    network_weights = d["network_weights"]

    model = Snrv(
        input_size,
        output_size,
        hidden_depth=hidden_depth,
        hidden_size=hidden_size,
        activation=activation,
        batch_norm=batch_norm,
        dropout_rate=dropout_rate,
        lr=lr,
        weight_decay=weight_decay,
        val_frac=val_frac,
        n_epochs=n_epochs,
        batch_size=batch_size,
        VAMPdegree=VAMPdegree,
        is_reversible=is_reversible,
    )

    model.evals = evals
    model.expansion_coefficients = expansion_coefficients
    model.load_weights(network_weights)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model
