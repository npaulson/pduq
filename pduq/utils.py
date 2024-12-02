"""
Utilities for PDUQ

Classes and functions defined here should have some reuse potential.
"""
import re

def database_symbols_to_fit(dbf, symbol_regex="^V[V]?([0-9]+)$"):
    """
    Return names of the symbols to fit that match the regular expression

    Parameters
    ----------
    dbf : Database
        pycalphad Database
    symbol_regex : str
        Regular expression of the fitting symbols. Defaults to V or VV followed by one or more numbers.

    Returns
    -------
    dict
        Context dictionary for different methods of calculation the error.
    """
    pattern = re.compile(symbol_regex)
    return sorted([x for x in sorted(dbf.symbols.keys()) if pattern.match(x)])