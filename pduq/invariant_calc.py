import logging
import numpy as np
from .dbf_calc import eq_calc_  # , get_eq_callables_
from espei.utils import database_symbols_to_fit
from pycalphad import variables as v
logging.basicConfig(filename='pduq.log', level=logging.INFO)


def invariant_samples(
        dbf, params, X, P, Tl, Tu, comp,
        client=None, comps=None, phases=None):
    """
    Find the composition and temperature of the invariants
    for parameter sets in params (for a binary)

    Parameters
    ----------

    dbf : Database
        Thermodynamic database containing the relevant parameters
    conds : dict or list of dict
        StateVariables and their corresponding value
    params : numpy array
        Array where the rows contain the parameter sets
        for the pycalphad equilibrium calculation
    X : float
        Guess for the mole fraction (of comp) of the invariant
    P : float
        Pressure (in Pa) at which to search for the invariants
    Tl : float
        Lower temperature bound to search for the invariants
    Tu : float
        Upper temperature bound to search for the invariants
    comp : str
        Name of the species
    client : Client, optional
        interface to dask.distributed compute cluster
    comps : list, optional
        Names of species to consider in the calculation
    phases : list or dict, optional
        Names of phases to consider in the calculation

    Returns
    -------
    Tv : list
        List of invariant temperatures corresponding to the
        parameter sets
    phv : list of list
        List of lists of phases
    bndv : numpy array
        Array where the first index corresponds to the parameter
        set, and the second index corresponds to the composition
        of the zero phase fraction bounaries of the first and last
        phases in phv, and of the three phase equilibrium.

    Examples
    --------
    >>> # let's do a multicore example
    >>> # first import modules and functions
    >>> import numpy as np
    >>> from dask.distributed import Client
    >>> from distributed.deploy.local import LocalCluster
    >>> from pycalphad import Database, variables as v
    >>> from pduq.invariant_calc import invariant_samples
    >>> # start the distributed client to parallelize the calculation
    >>> c = LocalCluster(n_workers=2, threads_per_worker=1)
    >>> client = Client(c)
    >>> # load the pycalphad database
    >>> dbf = Database('CU-MG_param_gen.tdb')
    >>> # load the parameter file
    >>> params = np.loadtxt('trace.csv', delimeter=',')[-2:, :]
    >>> # calculate the locations of invariant points for these two
    >>> # parameter sets in params
    >>> Tv, phv, bndv = invariant_samples(
    >>>     dbf, params, client=client, X=.2, P=101325,
    >>>     Tl=600, Tu=1400, comp='MG')
    >>> # print the temperatures of the invariant points
    >>> print(Tv)
    [1008.29467773 993.89038086]
    >>> # print the phases in equilibrium at the invariant points
    >>> print(phv)
    [['FCC_A1' 'LIQUID' 'LAVES_C15'], ['FCC_A1' 'LIQUID' 'LAVES_C15']]
    >>> # print the Mg molar fractions for the left phase boundary,
    >>> # the invariant, and the right phase boundary
    >>> print(bndv)
    [[0.04005779 0.21173958 0.33261747]
     [0.04096447 0.21720666 0.33295817]]
    """

    if comps is None:
        comps = list(dbf.elements)

    if phases is None:
        phases = list(dbf.phases.keys())

    neq = params.shape[0]  # calculate invariants for neq parameter sets

    symbols_to_fit = database_symbols_to_fit(dbf)

    # eq_callables = get_eq_callables_(dbf, comps, phases, symbols_to_fit)
    eq_callables = None  # eq_callables is disabled for current pycalcphad

    kwargs = {'dbf': dbf, 'comps': comps, 'phases': phases,
              'X': X, 'P': P, 'Tl': Tl, 'Tu': Tu, 'comp': comp,
              'params': params, 'symbols_to_fit': symbols_to_fit,
              'eq_callables': eq_callables}

    # invariant_(0, **kwargs)

    # define the map for the invariant calculation for neq parameter sets
    if client is None:
        invL = []
        for ii in range(neq):
            invL.append(invariant_(ii, **kwargs))
    else:
        A = client.map(invariant_, range(neq), **kwargs)
        invL = client.gather(A)
        client.close()

    # collect the key results after the map
    Tv = np.zeros((neq,))
    phv = neq*[None]
    bndv = np.zeros((neq, 3))
    for ii in range(neq):
        Tv[ii] = invL[ii][0]
        phv[ii] = invL[ii][1]
        bndv[ii, :] = invL[ii][2]

    return Tv, phv, bndv


def invariant_(index, dbf, params, comps, phases, X, P, Tl, Tu, comp,
               symbols_to_fit, eq_callables):

    kwargs = {'dbf': dbf, 'comps': comps, 'phases': phases,
              'paramA': params[index, :], 'symbols_to_fit': symbols_to_fit,
              'eq_callables': eq_callables}

    def mini(T):
        """
        perform the equilibrium calculation at a specified temperature
        and return the list of unique phases in equilibrium along with the
        equlibrium data object
        """
        conds = {v.P: P, v.T: T, v.X(comp): X}
        eq = eq_calc_(conds=conds, **kwargs)
        PhT = list(np.unique(eq.Phase))
        if '' in PhT:
            PhT.remove('')
        return eq, PhT

    # perform equilibrium calculations at the lower bound, middle, and
    # upper bound temperatures
    Tm = 0.5*(Tl+Tu)  # middle temperature
    eql, PhTl = mini(Tl)
    eqm, PhTm = mini(Tm)
    equ, PhTu = mini(Tu)

    # now we use the bisection method to find the invariant temperature
    # and composition

    # identify the number of calculations so that the error in the
    # temperature is less than errlim
    errlim = 0.01
    niter = np.log(errlim/(Tu-Tl))/np.log(0.5) - 1
    niter = np.int16(np.ceil(niter))

    for ii in range(niter):
        # if the phases in equilibrium at the middle temperature are
        # different than at the lower temperature, then the invariant
        # will be between the lower and middle temperature. We can then
        # set the upper temperature to the previous middle temperature.
        # Otherwise, the invariant will be between the middle and upper
        # temperature.
        if str(PhTm) != str(PhTl):
            Tu = Tm
            equ = eqm
            PhTu = PhTm
        else:
            Tl = Tm
            eql = eqm
            PhTl = PhTm
        Tm = 0.5*(Tl + Tu)  # get the new middle temperature
        eqm, PhTm = mini(Tm)  # calculate the Tm equilibrium
        # print(Tm, ' ', PhTm)

    def getbnd(eq, PhT):
        """
        get the molar compositions of the phases in PhT
        """
        bnd = []
        for phase in PhT:
            tmp = eq.X.where(eq.Phase == phase)
            tmp = tmp.sel(component=comp).sum(dim='vertex')
            bnd.append(np.squeeze(tmp.values))
        return np.array(bnd)

    # PhTA are the phases at Tl and Tu
    PhTA = np.concatenate([PhTl, PhTu])
    # bndA are the molar compositions of the phases in PhTl and PhTu
    bndA = np.concatenate([getbnd(eql, PhTl), getbnd(equ, PhTu)])
    # phs are the unique phases in the three phase region
    phs, indx = np.unique(PhTA, return_index=True)
    # bnd are the molar compositions corresponding to phs
    bnd = bndA[indx]

    indx = np.argsort(bnd)
    phs = list(phs[indx])
    bnd = bnd[indx]

    logging.info('Invariant computed for set ' + str(index))
    logging.info('T = ' + str(Tm) + ' +/- ' + str(Tm-Tl) + 'K')
    logging.info(str(phs))
    logging.info(str(bnd))

    # print('Invariant computed for set ', index)
    # print('T=', Tm, '+/-', Tm-Tl, 'K')
    # print(phs)
    # print(bnd)

    return Tm, phs, bnd, index
