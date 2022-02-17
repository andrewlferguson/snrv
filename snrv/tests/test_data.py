import torch
import snrv.data


def test_DatasetSnrv():

    N = 100
    lag = 10

    data = torch.randn(N)
    ln_dynamical_weight = torch.randn(N)
    thermo_weight = torch.exp(torch.randn(N))
    dataset = snrv.data.DatasetSnrv(data, lag, ln_dynamical_weight, thermo_weight)

    assert len(dataset) == (N - lag)

    n = 8

    data = [torch.randn(N) for _ in range(n)]
    ln_dynamical_weight = [torch.randn(N) for _ in range(n)]
    thermo_weight = [torch.exp(torch.randn(N)) for _ in range(n)]
    dataset = snrv.data.DatasetSnrv(data, lag, ln_dynamical_weight, thermo_weight)

    assert len(dataset) == (n * (N - lag))
