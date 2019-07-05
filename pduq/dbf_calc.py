import logging
import pickle
import sympy
import numpy as np
import xarray as xr
from collections import OrderedDict
from espei.utils import database_symbols_to_fit
from itertools import chain
from pycalphad import equilibrium, variables as v
from pycalphad.codegen.callables import build_callables
from pycalphad.core.utils import instantiate_models
from time import time
logging.basicConfig(filename='pduq.log', level=logging.INFO)


def eq_calc_(dbf, comps, phases, conds,
             paramA, symbols_to_fit,
             eq_callables=None):

    param_dict = {param_name: param for param_name, param
                  in zip(symbols_to_fit, paramA)}

    parameters = OrderedDict(sorted(param_dict.items(), key=str))

    eq_result = equilibrium(dbf, comps, phases, conds,
                            parameters=parameters, callables=eq_callables)

    return eq_result


def eq_calc_chunk_(chunk, dbf, comps, phases, conds,
                   params, symbols_to_fit,
                   eq_callables=None):
    """
    Perform equilibrium calculations for the list of indices
    in chunk corresponding the the parameter sets in params

    Parameters
    ----------
    chunk : list of int
        List of indices corresponding to first axis in params
    dbf : Database
        Thermodynamic database containing the relevant parameters
    comps : list
        Names of components to consider in the calculation
    phases : list or dict
        Names of phases to consider in the calculation
    conds: dict or (list of dict)
        StateVariables and their corresponding value
    params : numpy array
        Array where the rows contain the parameter sets
        for the pycalphad equilibrium calculation
    symbols_to_fit : list of str
        List of symbols in the Database that will be fit. If
        None (default) are passed, then all parameters
        prefixed with VV followed by a number, e.g.
        VV0001 will be fit
    eq_callables : dict, optional
        Pre-computed callable functions for equilibrium calculation

    Returns
    -------
    list of structured equilibrium calculations

    Examples
    --------
    None yet
    """

    eq_result = []

    for index in chunk:

        paramA = np.squeeze(params[index, :])

        # param_dict = {param_name: param for param_name, param
        #               in zip(symbols_to_fit, np.squeeze(params[index, :]))}

        # parameters = OrderedDict(sorted(param_dict.items(), key=str))

        # eq_result_ = equilibrium(dbf, comps, phases, conds,
        #                          parameters=parameters, callables=eq_callables)

        eq_result_ = eq_calc_(dbf, comps, phases, conds,
                              paramA, symbols_to_fit, eq_callables)

        eq_result += [eq_result_]

    logging.info(str(chunk) + ' ' + str(time()))

    return eq_result


def eq_calc_samples(
        client, dbf, conds, params, comps=None, phases=None,
        savef=None):
    """
    Perform equilibrium calculations for the parameter sets
    in params

    Parameters
    ----------
    client : Client
        interface to dask.distributed compute cluster
    dbf : Database
        Thermodynamic database containing the relevant parameters
    conds: dict or (list of dict)
        StateVariables and their corresponding value
    params : numpy array
        Array where the rows contain the parameter sets
        for the pycalphad equilibrium calculation
    comps : list
        Names of components to consider in the calculation
    phases : list or dict
        Names of phases to consider in the calculation
    savef : string
        Save file for the equilibrium calculations

    Returns
    -------
    structured equilibrium calculations for parameter sets in
    params

    Examples
    --------
    None yet
    """

    if comps is None:
        comps = list(dbf.elements)

    if phases is None:
        phases = list(dbf.phases.keys())

    symbols_to_fit = database_symbols_to_fit(dbf)

    # eq_callables = get_eq_callables_(dbf, comps, phases, symbols_to_fit)
    eq_callables = None

    kwargs = {'dbf': dbf, 'comps': comps, 'phases': phases, 'conds': conds,
              'params': params, 'symbols_to_fit': symbols_to_fit,
              'eq_callables': eq_callables}

    neq = params.shape[0]

    if neq < 20:
        nch = neq
    else:
        nch = 20
    chunks = [list(range(neq))[ii::nch] for ii in range(nch)]

    A = client.map(eq_calc_chunk_, chunks, **kwargs)

    eqL = client.gather(A)

    eqL = list(chain.from_iterable(eqL))

    client.close()

    eqC = xr.concat(eqL, 'sample')
    eqC.coords['sample'] = np.arange(neq)

    logging.info(str(eqC))

    if savef is not None:
        with open(savef, 'wb') as buff:
            pickle.dump(eqC, buff)

    return eqC


def get_eq_callables_(dbf, comps, phases, symbols_to_fit):

    for x in symbols_to_fit:
        if isinstance(dbf.symbols[x], sympy.Piecewise):
            dbf.symbols[x] = dbf.symbols[x].args[0].expr

    models = instantiate_models(
        dbf, comps, phases,
        parameters=dict(zip(symbols_to_fit, [0]*len(symbols_to_fit))))

    eq_callables = build_callables(
        dbf, comps, phases, models,
        parameter_symbols=symbols_to_fit,
        build_gradients=True, build_hessians=False,
        additional_statevars={v.N, v.P, v.T})

    logging.info('callables have been created')

    return eq_callables
