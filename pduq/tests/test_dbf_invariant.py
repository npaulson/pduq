import numpy as np
from pkg_resources import resource_filename
from pycalphad import Database, variables as v
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

    assert list(eqC.dims.values()) == [1, 1, 1, 1, 2, 4, 2, 3]
    assert list(np.squeeze(tst)) == [0., 1.]


def test_invariant_samples():

    tdbfile = resource_filename('pduq.tests', 'CU-MG_param_gen.tdb')
    dbf = Database(tdbfile)

    paramfile = resource_filename('pduq.tests', 'trace.csv')
    params = np.loadtxt(paramfile, delimiter=',')

    Tv, phv, bndv = invariant_samples(
        dbf, params[-2:, :],
        X=.2, P=101325, Tl=600, Tu=1400, comp='MG')

    Tv_ref = [1008.29467773, 993.89038086]
    phv_ref = [['FCC_A1', 'LIQUID', 'LAVES_C15'],
               ['FCC_A1', 'LIQUID', 'LAVES_C15']]
    bndv_ref = [[0.04005779, 0.21173958, 0.33261747],
                [0.04096447, 0.21720666, 0.33295817]]

    assert np.all(np.isclose(Tv, Tv_ref, atol=1e-5))
    assert np.all(phv == phv_ref)
    assert np.all(np.isclose(bndv, bndv_ref, atol=1e-5))
