import numpy as np
import pycalphad.variables as v
from pkg_resources import resource_filename
from pycalphad import Database
from pduq.dbf_calc import eq_calc_samples
from pduq.invariant_calc import invariant_samples


def test_eq_calc_samples():

    tdbfile = resource_filename('pduq.tests', 'CU-MG_param_gen.tdb')
    dbf = Database(tdbfile)

    paramfile = resource_filename('pduq.tests', 'trace.csv')
    params = np.loadtxt(paramfile, delimiter=',')

    conds = {v.P: 101325, v.T: 1003, v.X('MG'): 0.214}

    eqC = eq_calc_samples(dbf, conds, params[-2:, :])

    tst = eqC.NP.where(eqC.Phase == 'LIQUID').sum(dim='vertex')

    assert eqC.dims == {'sample': 2, 'N': 1, 'P': 1, 'T': 1, 'X_MG': 1,
        'vertex': 3, 'component': 2, 'internal_dof': 4}

    assert list(np.squeeze(tst).values) == [0., 1.]


def test_invariant_samples():

    tdbfile = resource_filename('pduq.tests', 'CU-MG_param_gen.tdb')
    dbf = Database(tdbfile)

    paramfile = resource_filename('pduq.tests', 'trace.csv')
    params = np.loadtxt(paramfile, delimiter=',')

    Tv, phv, bndv = invariant_samples(
        dbf, params[-2:, :],
        X=.2, P=101325, Tl=600, Tu=1400, comp='MG')

    Tv_ref = [1008.35571289, 993.95141602]
    phv_ref = [['FCC_A1', 'LIQUID', 'LAVES_C15'],
               ['FCC_A1', 'LIQUID', 'LAVES_C15']]
    bndv_ref = [[0.04006727, 0.21171107, 0.33261726],
                [0.04097448, 0.21717864, 0.33295806]]

    assert np.all(np.isclose(Tv, Tv_ref, atol=1e-5))
    assert np.all(phv == phv_ref)
    assert np.all(np.isclose(bndv, bndv_ref, atol=1e-5))


if __name__ == '__main__':

    test_eq_calc_samples()
    test_invariant_samples()
