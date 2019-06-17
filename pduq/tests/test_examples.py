import numpy as np
from pkg_resources import resource_filename
from pycalphad import Database, variables as v
from dask.distributed import Client
from distributed.deploy.local import LocalCluster
from pduq.dbf_calc import eq_calc_samples


def test_eq_calc_samples():

    c = LocalCluster(n_workers=2, threads_per_worker=1)
    client = Client(c)

    tdbfile = resource_filename('pduq.tests', 'CU-MG_param_gen.tdb')
    dbf = Database(tdbfile)

    paramfile = resource_filename('pduq.tests', 'trace.csv')
    params = np.loadtxt(paramfile, delimiter=',')

    conds = {v.P: 101325, v.T: 1003, v.X('MG'): 0.214}

    eqC = eq_calc_samples(client, dbf, conds, params[-2:, :])

    tst = eqC.NP.where(eqC.Phase == 'LIQUID').sum(dim='vertex')

    assert list(eqC.dims.values()) == [1, 1, 1, 1, 2, 4, 2, 3]

    assert list(np.squeeze(tst)) == [0., 1.]
