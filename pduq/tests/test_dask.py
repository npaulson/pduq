import numpy as np
import pycalphad.variables as v
from dask.distributed import Client, LocalCluster
from pycalphad import Database
from pduq.dbf_calc import eq_calc_samples
from pduq.uq_plot import get_phase_prob, plot_dist
from pkg_resources import resource_filename


def test_eq_calc_samples():
    tdbfile = resource_filename('pduq.tests', 'CU-MG_param_gen.tdb')
    dbf = Database(tdbfile)
    
    paramfile = resource_filename('pduq.tests', 'trace.csv')
    params = np.loadtxt(paramfile, delimiter=',')
    
    c = LocalCluster(n_workers=2, threads_per_worker=1)
    client = Client(c)
    
    # Define the equilibrium conditions including pressure (Pa), temperature (K), and molar composition Mg
    conds = {v.P: 101325, v.T: 1003, v.X('MG'): 0.214}
    
    # Perform the equilibrium calculation for all parameter sets
    eqC = eq_calc_samples(dbf, conds, params, client=client)

    # Check that the dimensions of eqC match the expected values
    assert eqC.dims == {
        'sample': 10, 
        'N': 1, 
        'P': 1, 
        'T': 1, 
        'X_MG': 1,
        'vertex': 3, 
        'component': 2, 
        'internal_dof': 4
    }

    # Clean up the Dask client
    client.close()
