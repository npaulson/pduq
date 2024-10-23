import matplotlib.pyplot as plt
from pycalphad import calculate
import numpy as np
import seaborn as sns
import xarray as xr
# from espei.core_utils import get_data
from espei.utils import database_symbols_to_fit
from collections import OrderedDict
from scipy.stats import gaussian_kde

sns.set_theme(color_codes=True)


def get_label(cplt):

    """
    get cplt and return a label

    Parameters
    ----------
    cplt : str
        name of the dimension of interest

    Returns
    -------
    label : str
        axis label

    Examples
    --------
    None yet
    """
    label = cplt
    if cplt == 'T':
        label += ' (K)'
    elif cplt == 'P':
        label += ' (Pa)'
    return label


def get_phase_prob(eq, phaseregL):

    """
    Get the probablities for the presence of the desired phase region
    specified by phaseregL.

    Parameters
    ----------
    eq : xarray object
        Structured equilibirum calculation containing a 'sample'
        dimension correspoinding to different parameter sets
    phaseregL : tuple or list of str
        list of considered phases in equilibirum

    Returns
    -------
    prob : array
        Probabilities of non-zero phase fraction for the phase
        region of interest in the shape of the conditions for the
        equilibrium calculation.

    Examples
    --------
    >>> import pickle
    >>> import pduq.uq_plot as uq
    >>> # load the collated equilibrium calculation for a single XTP
    >>> # point as produced by dbf_calc.eq_calc_samples
    >>> with open('single_point.pkl', 'rb') as buff:
    >>>     eq = pickle.load(buff)
    >>> # define a set of phases in equilibrium to evaluate
    >>> phaseregL = ['FCC_A1', 'LIQUID']
    >>> # calculate the probability of the set of phases having a
    >>> # non-zero phase fraction
    >>> print(uq.get_phase_prob(eq, phaseregL))
    0.2
    """

    phsum = np.zeros(eq.NP.shape[:-1])
    phpres = np.ones(eq.NP.shape[:-1])
    # loop over phase labels in phaseregL
    for phase in phaseregL:
        # get the phase fraction for the desired phase
        NP_ = eq.NP.where(eq.Phase == phase)
        # sum over vertex
        NP_ = NP_.sum(dim='vertex')
        phsum += NP_
        phpres *= NP_ > 0
    # for a particular sample a phase region is present
    # if the phases of interest sum to one and if both
    # phases are present
    ineq = (phsum > 1 - 1e-6)*phpres
    # identify what fraction of samples have the phase
    # present in X-T-P space
    prob = np.mean(ineq, 0)

    return prob


def get_ticks(eq, cplt):

    """
    get tick locations and values

    Parameters
    ----------
    eq : xarray object
        Structured equilibirum calculation
    cplt : str
        name of the dimension of interest

    Returns
    -------
    ticpts : numpy array
        array of tick locations
    ticvals : numpy array
        array of tick values

    Examples
    --------
    None yet
    """

    ticvalall = eq.get(cplt).values  # coordinates along cplt

    if cplt == 'T' or cplt == 'P':
        ticvalall = np.round(ticvalall, 0)
    else:
        ticvalall = np.round(ticvalall, 2)
    # get roughly 6 tick locations
    ntic = len(ticvalall)
    ticpts = np.arange(
        0, ntic, np.int32(np.floor(ntic/6)))
    ticvals = ticvalall[ticpts]
    return ticpts, ticvals


def plot_dist(eq, coordD, phaseregL, phase, typ, figsize=None):

    """
    Plot the distribution of a property for all parameter sets
    where the phases of interest are in equilibrium.

    Parameters
    ----------
    eq : xarray object
        Structured equilibirum calculation containing a 'sample'
        dimension correspoinding to different parameter sets
    coorD : dict
        Dictionary with 'T' for temperature, 'X_EL' for the molar
        composition of element EL, and 'component' for the element
        to consider for the composition
    phaseregL : tuple or list of str
        list of considered phases in equilibirum
    phase : str
        Phase of interest. This must be specified, but only
        impacts the calculation for the NP and X properties
    typ : str
        The quantity to plot. Available options are:
        NP: phase fraction
        X: molar composition of the selected component
        GM: molar Gibbs energy of the X-T-P point
        MU: chemical potential of the selected component
    figsize : tuple or list of int or float, optional
        Plot dimensions in inches

    Returns
    -------
    compL : numpy array
        1D array with typ values for all parameter sets where
        only the phases in phaseregL are in equilibrium

    Examples
    --------
    >>> import pickle
    >>> import pduq.uq_plot as uq
    >>> # load the collated equilibrium calculation for a single XTP
    >>> # point as produced by dbf_calc.eq_calc_samples
    >>> with open('single_point.pkl', 'rb') as buff:
    >>>     eq = pickle.load(buff)
    >>> # identify the XTP point of interest
    >>> coordD = {'T':1003, 'X_MG':.214, 'component':'MG'}
    >>> phaseregL = ['FCC_A1', 'LIQUID']
    >>> phase = 'FCC_A1'
    >>> # plot the distribution of phase fractions for the selected
    >>> # phase in an equilibrium calculation with the phases
    >>> # considered in phaseregL.
    >>> uq.plot_dist(eq, coordD, phaseregL, phase, typ='NP')
    """

    compL = np.array([])  # array to collect property of interest
    for ii in range(eq.sizes['sample']):
        # eq_: equilibrium object for sample ii
        eq_ = eq.sel({'sample': ii}).sel(coordD)

        # phaseregL_: phases present in eq_
        phaseregL_ = list(np.squeeze(eq_.Phase.values))
        if '' in phaseregL_:
            phaseregL_.remove('')
        phaseregL.sort()
        phaseregL_.sort()

        # only include information from eq_ in final distribution
        # if the phases in eq_ are the same as the desired phase-
        # region in phaseregL
        if phaseregL == phaseregL_:
            val = eq_[typ].where(eq_.Phase == phase)
            val = val.sum(dim='vertex').values[0]
            # append the property of interest to compL
            compL = np.append(compL, val)

    plt.figure(figsize=figsize)
    sns.histplot(compL, stat='density', kde=True)
    xlabeld = {'NP': '%s phase fraction' % phase,
               'X': r'$\mathrm{x_{%s}}$' % coordD['component'],
               'GM': 'Molar Gibbs energy',
               'MU': 'chemical potential, %s' % coordD['component']}
    plt.xlabel(xlabeld[typ], fontsize='large')
    plt.ylabel('frequency', fontsize='large')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    return compL


def plot_property(dbf, comps, phaseL, params, T, prop,
                  config=None, datasets=None, xlim=None,
                  xlabel=None, ylabel=None, yscale=None,
                  phase_label_dict=None,
                  unit='kJ/mol.', cdict=None, figsize=None):

    """
    Plot a property of interest versus temperature with uncertainty
    bounds for all phases of interest

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters
    comps : list
        Names of components to consider in the calculation
    phaseL : list
        Names of phases to plot properties for
    params : numpy array
        Array where the rows contain the parameter sets
        for the pycalphad equilibrium calculation
    T : list, array or x-array object
        Temperature values at which to plot the selected property
    prop : str
        property (or attribute in pycalphad terminology) to sample,
        e.g. GM for molar gibbs energy or H_MIX for the enthalpy of
        mixing
    config : tuple, optional
        Sublattice configuration as a tuple, e.g. (“CU”, (“CU”, “MG”))
    datasets : espei.utils.PickleableTinyDB, optional
        Database of datasets to search for data
    xlims : list or tuple of float, optional
        List or tuple with two floats corresponding to the
        minimum and maximum molar composition of comp
    xlabel : str, optional
        plot x label
    ylabel : str, optional
        plot y label
    yscale : int or float, optional
        scaling factor to apply to property (e.g. to plot kJ/mol.
        instead of J/mol. choose yscale to be 0.001)
    phase_label_dict : dict, optional
        Dictionary with keys given by phase names and corresponding
        strings to use in plotting (e.g. to enable LaTeX labels)
    unit : str, optional
        Unit to plot on the y-axis for the property of interest
    cdict : dict, optional
        Dictionary with phase names and corresponding
        colors
    figsize : tuple or list of int or float, optional
        Plot dimensions in inches

    Returns
    -------

    Examples
    --------
    >>> import numpy as np
    >>> import pduq.uq_plot as uq
    >>> from pycalphad import Database
    >>> dbf = Database('CU-MG_param_gen.tdb')
    >>> comps = ['MG', 'CU', 'VA']
    >>> phaseL = ['CUMG2', 'LIQUID']
    >>> params = np.loadtxt('params.npy')[: -1, :]
    >>> T = 650
    >>> prop = 'GM'
    >>> # Plot the molar gibbs energy of all phases in phaseL
    >>> # versus molar fraction of MG at 650K. This will have
    >>> # uncertainty intervals generated by the parameter sets
    >>> # in params
    >>> uq.plot_property(dbf, comps, phaseL, params, T, prop)
    """

    symbols_to_fit = database_symbols_to_fit(dbf)

    CI = 95
    nph = len(phaseL)
    colorL = sns.color_palette("cubehelix", nph)
    markerL = 10*['o', 'D', '^', 'x', 'h', 's',
                  'v', '*', 'P', 'p', '>', 'd', '<']

    plt.figure(figsize=figsize)

    # compute uncertainty in property for each phase in list
    for ii in range(nph):
        phase = phaseL[ii]
        print('starting', prop, 'evaluations for the', phase, 'phase')

        # for each parameter sample calculate the property
        # for each possible site occupancy ratios
        compL = []
        for index in range(params.shape[0]):
            param_dict = {param_name: param for param_name, param
                          in zip(symbols_to_fit, params[index, :])}
            parameters = OrderedDict(sorted(param_dict.items(), key=str))
            comp = calculate(
                dbf, comps, phase,
                P=101325, T=T, output=prop, parameters=parameters)
            compL += [comp]

        # concatenate the calculate results in an xarray along
        # an axis named 'sample'
        compC = xr.concat(compL, 'sample')
        compC.coords['sample'] = np.arange(params.shape[0])

        # The composition vector is the same for all samples
        if hasattr(T, "__len__"):
            Xvals = T
        else:
            Xvals = comp.X.sel(component=comps[0]).values.squeeze()
        Pvals = compC[prop].where(compC.Phase == phase).values.squeeze()

        if np.array(Xvals).size == 1:
            print('phase is a line compound')
            Xvals_ = np.array([Xvals-0.002, Xvals+0.002])
            Pvals_ = np.vstack([Pvals, Pvals]).T
        else:
            # find the lower hull of the property by finding
            # the configuration with the lowest value within
            # each interval. In each interval record the composition
            # and property
            indxL = np.array([])
            # Xbnds = np.arange(0, 1.01, 0.01)
            Xbnds = np.linspace(Xvals.min(), Xvals.max(), 100)
            for lb, ub in zip(Xbnds[:-1], Xbnds[1:]):
                # print('lb: ', lb, ', ub: ', ub)
                boolA = (lb <= Xvals)*(Xvals < ub)
                if boolA.sum() == 0:
                    continue
                indxA = np.arange(boolA.size)[boolA]
                P_ = Pvals[0, boolA]
                indxL = np.append(indxL, indxA[P_.argmin()])
                # indxL = np.append(indxL, indxA[P_.argmax()])
            indxL = indxL.astype('int32')

            if indxL.size == 1:
                print('only one point found')
                Xvals_ = Xvals[np.asscalar(indxL)]
                Pvals_ = Pvals[:, np.asscalar(indxL)]
            else:
                Xvals_ = Xvals[indxL]
                Pvals_ = Pvals[:, indxL]

        # Xvals_ = Xvals
        # Pvals_ = Pvals
        # for ii in range(params.shape[0]):
        #     plt.plot(Xvals_, Pvals_[ii, :], 'k-', linewidth=0.5, alpha=0.1)
        # plt.show()

        if yscale is not None:
            Pvals_ *= yscale

        low, mid, high = np.percentile(
            Pvals_, [0.5*(100-CI), 50, 100-0.5*(100-CI)], axis=0)

        if cdict is not None:
            color = cdict[phase]
        else:
            color = colorL[ii]

        if phase_label_dict is not None:
            label = phase_label_dict[phase]
        else:
            label = phase

        plt.plot(Xvals_, mid, linestyle='-', color=color, label=label)
        plt.fill_between(
            np.atleast_1d(Xvals_), low, high, alpha=0.3, facecolor=color)

        # collect and plot experimental data
        if config is not None and datasets is not None:
            symmetry = None
            data = get_data(
                comps, phase, config, symmetry, datasets, prop)
            print(data)
            for data_s, marker in zip(data, markerL):
                occupancies = data_s['solver']['sublattice_occupancies']
                # at the moment this needs to be changed manually
                X_vec = [row[0][0] for row in occupancies]
                values = np.squeeze(data_s['values'])

                if yscale is not None:
                    values *= yscale

                plt.plot(
                    X_vec, values, linestyle='', marker=marker,
                    markerfacecolor='none', markeredgecolor=color,
                    markersize=6, alpha=0.9, label=data_s['reference'])

    if xlim is None:
        plt.xlim([Xvals_.min(), Xvals_.max()])
    else:
        plt.xlim(xlim)

    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(r'$X_{%s}$' % comps[0])

    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(prop + ' (' + unit + ')')

    plt.legend()
    plt.tight_layout()


def plot_binary(eq, comp, alpha=None, cdict=None):

    """
    Plot a binary phase diagram. This purposefully has a
    minimal number of options so that the returned figure
    can be customized easily.

    Parameters
    ----------
    eq : xarray object
        Structured equilibirum calculation
    comp : str
        Label for species to plot on the x-axis,
        e.g. MG for magnesium
    alpha : float, optional
        Number between 0 and 1 for the line transparency
    cdict : dict, optional
        Dictionary with phase names and corresponding
        colors

    Returns
    -------
    compL : numpy array
        1D array with typ values for all parameter sets where
        only the phases in phaseregL are in equilibrium

    Examples
    --------
    >>> import pickle
    >>> import pduq.uq_plot as uq
    >>> with open('single.pkl', 'rb') as buff:
    >>>     eq = pickle.load(buff)
    >>> comp = 'MG'
    >>> # plot a binary phase diagram for a set of
    >>> # equilibrium calculations, and comp as the
    >>> # molar fraction on the x-axis
    >>> uq.plot_binary(eq, comp)
    """

    Tvec = eq.get('T').values  # all temperature values
    Xvec = eq.get('X_' + comp).values  # all molar composition values

    # phaseL: list of unique phases
    phaseL = list(np.unique(eq.get('Phase').values))
    if '' in phaseL:
        phaseL.remove('')
    nph = len(phaseL)  # number of unique phases

    Xph = np.zeros((Tvec.size, Xvec.size, nph))

    for ii in range(nph):
        tmp = eq.X.where(eq.Phase == phaseL[ii])
        tmp = tmp.sel(component=comp)

        nans = np.squeeze(tmp.values)
        nans = np.isnan(nans[..., 0])*np.isnan(nans[..., 1])
        Xph_ = np.squeeze(tmp.sum(dim='vertex').values)
        Xph_[nans] = np.nan
        Xph[..., ii] = Xph_

    XphD = {}

    for ii in range(Tvec.size):
        T = str(np.int32(Tvec[ii]))
        xl = 'X_' + T
        pl = 'Ph_' + T
        XphD[xl], XphD[pl] = [], []
        for jj in range(nph):
            # find values not equal to the composition
            neq2comp = np.isclose(Xph[ii, :, jj], Xvec, atol=1e-4)
            neq2comp = np.invert(neq2comp)
            vals = Xph[ii, neq2comp, jj]

            # remove nans
            vals = vals[np.invert(np.isnan(vals))]

            # get unique values
            vals_ = np.round(vals, 5)
            loc, indx = np.unique(vals_, return_index=True)
            vals = vals[indx]

            XphD[xl] += list(vals)
            XphD[pl] += len(vals)*[phaseL[jj]]

        arg = np.argsort(np.array(XphD[xl]))
        XphD[xl] = np.array(XphD[xl])[arg]
        XphD[pl] = np.array(XphD[pl])[arg]

    colorL = sns.color_palette("cubehelix", nph)
    if alpha is None:
        alpha = 0.9

    for ii in range(Tvec.size):
        T = str(np.int32(Tvec[ii]))
        xl = 'X_' + T
        pl = 'Ph_' + T
        vals = XphD[xl]
        phs = XphD[pl]
        for jj in range(nph):
            if phaseL[jj] not in list(phs):
                continue
            vals_ = vals[phs == phaseL[jj]]

            if cdict is not None:
                color = cdict[phaseL[jj]]
            else:
                color = colorL[jj]

            plt.plot(vals_, [Tvec[ii]]*len(vals_),
                     marker='o', markersize=2, alpha=alpha,
                     color=color, linestyle='', mew=0.0)


def plot_contour(points, c='k', bw=.3):

    """
    Plot as set of KDE probability density contours for a set
    of points, typically corresponding to invariant locations

    Parameters
    ----------
    points : numpy array
        an array of compositions and temperatures representing
        invariant points or some other phase diagram feature
    c : color, optional
        color of the density contours
    bw : float, optional
        KDE bandwidth

    Returns
    -------

    Examples
    --------
    >>> import numpy as np
    >>> from pduq.invariant_calc import invariant_samples
    >>> from pduq.uq_plot import plot_contour
    >>> from pycalphad import Database
    >>> # load dbf file and raw parameter set.
    >>> dbf = Database('CU-MG_param_gen.tdb')
    >>> params = np.loadtxt('trace.csv', delimiter=',')
    >>> # find the set of invariant points
    >>> Tv, phv, bndv = invariant_samples(
    >>>     dbf, params, X=.2, P=101325, Tl=600, Tu=1400,
    >>>     comp='MG')
    >>> # define the 'points' array
    >>> points = np.zeros((len(Tv, 2)))
    >>> points[:, 0] = bndv[:, 1]
    >>> points[:, 1] = Tv
    >>> # plot the contour
    >>> plt.figure()
    >>> uq.plot_contour(points)
    """
    kernel = gaussian_kde(points.T)

    buf_mult = 1.5

    xmin = points[:, 0].min()
    xmax = points[:, 0].max()
    buf = buf_mult*(xmax-xmin)
    xvec = np.linspace(xmin-buf, xmax+buf, 100)

    ymin = points[:, 1].min()
    ymax = points[:, 1].max()
    buf = buf_mult*(ymax-ymin)
    yvec = np.linspace(ymin-buf, ymax+buf, 100)

    X, Y = np.meshgrid(xvec, yvec)
    XY = np.vstack([X.ravel(), Y.ravel()])

    kernel = gaussian_kde(points.T, bw_method=bw)
    pdf = kernel(XY).T
    Z = pdf.reshape(X.shape)

    order = pdf.argsort()
    pdf_s = pdf[order]
    F = np.cumsum(pdf_s)
    F /= F[-1]

    Plvls = np.array([1-.9545, 1-.6827])
    boundaries = F.searchsorted(Plvls)
    lvls = pdf_s[boundaries]

    CS = plt.contour(X, Y, Z, lvls, colors=c, alpha=.7, linestyles=[':', '-'])
    labels = ["95% CI", "68% CI"]

    for ii in range(len(labels)):
        CS.collections[ii].set_label(labels[ii])


def plot_phasefracline(eq, coordD, xlabel=None,
                       phase_label_dict=None, cdict=None, figsize=None):

    """
    Plot the phase fraction with uncertainty versus composition,
    temperature or pressure.

    Parameters
    ----------
    eq : xarray object
        Structured equilibirum calculation containing a 'sample'
        dimension correspoinding to different parameter sets
    coordD : dict
        Dictionary defining constraints on the coordinates in
        eq for plotting the phase fraction with varying X, T or P
        For example, we might pick a fixed X and let T vary
        as follows: coordD = {'X_MG':0.1}
    xlabel : str, optional
        Label for the x-axis
    phase_label_dict : dict, optional
        Dictionary with keys given by phase names and corresponding
        strings to use in plotting (e.g. to enable LaTeX labels)
    xlims : list or tuple of float, optional
        List or tuple with two floats corresponding to the
        minimum and maximum molar composition of comp
    cdict : dict, optional
        Dictionary with phase names and corresponding
        colors
    figsize : tuple or list of int or float, optional
        Plot dimensions in inches

    Returns
    -------

    Examples
    --------
    >>> import pickle
    >>> import pduq.uq_plot as uq
    >>> with open('full.pkl', 'rb') as buff:
    >>>     eq = pickle.load(buff)
    >>> # if for 'full.pkl' X_MG and T have 100 intervals each
    >>> # we can fix T and plot the phase fraction versus
    >>> # X_MG with uncertainty
    >>> coordD = {'T':1000}
    >>> # plot phase fraction versus X_MG
    >>> uq.plot_phasefracline(eq, coordD, xlabel='X_MG')
    """
    phaseL = list(np.unique(eq.get('Phase').values))
    if '' in phaseL:
        phaseL.remove('')
    nph = len(phaseL)

    eq = eq.sel(coordD)

    # the X, T or P dimension with the most values
    # is then plotted on the x-axis
    rmlist = ['N', 'internal_dof', 'sample', 'vertex']
    max_sz = 0
    for dim in list(eq.dims):
        if dim not in rmlist:
            dim_sz = eq.sizes[dim]
            if dim_sz > max_sz:
                max_sz = dim_sz
                dim_max = dim
    xvec = eq.get(dim_max).values

    CI = 95
    colorL = sns.color_palette("cubehelix", nph)

    plt.figure(figsize=figsize)

    for ii in range(nph):
        phase = phaseL[ii]
        val = eq.NP.where(eq.Phase == phase)
        val = val.sum(dim='vertex').values.squeeze()

        low, mid, high = np.percentile(
            val, [0.5*(100-CI), 50, 100-0.5*(100-CI)], axis=0)

        if phase_label_dict is not None:
            label = phase_label_dict[phase]
        else:
            label = phase

        if cdict is not None:
            color = cdict[phaseL[ii]]
        else:
            color = colorL[ii]

        plt.plot(xvec, mid, linestyle='-', color=color, label=label)
        plt.fill_between(
            np.atleast_1d(xvec), low, high, alpha=0.3, facecolor=color)

    plt.xlim([xvec.min(), xvec.max()])
    plt.ylim([-0.01, 1.01])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(xlabel, fontsize='large')
    plt.ylabel('phase fraction', fontsize='large')
    plt.legend()
    plt.tight_layout()


def plot_phasereg_prob(eq, phaseregL, title=None, figname=None,
                       coordplt=['X', 'T'], typ='grayscale', figsize=None):

    """
    Plot the probabilty of non-zero phase fraction for a combination of
    phases in equilibrium versus composition and temperature.

    Parameters
    ----------
    eq : xarray object
        Structured equilibirum calculation containing a 'sample'
        dimension correspoinding to different parameter sets
    phaseregL : tuple or list of str
        list of considered phases in equilibirum
    title : str, optional
        title of the plot
    figname : str, optional
        name of the figure to differentiate plot windows
    coordplt : tuple or list of str, optional
        list containing names of the axes
    typ : str, optional
        plot type. This can either be 'grayscale' or 'contour'
    figsize : tuple or list of int or float, optional
        Plot dimensions in inches

    Returns
    -------

    Examples
    --------
    >>> import pickle
    >>> import pduq.uq_plot as uq
    >>> with open('multiple.pkl', 'rb') as buff:
    >>>     eq = pickle.load(buff)
    >>> phaseregL = ['FCC_A1', 'LIQUID']
    >>> # plot the probability of non-zero phase
    >>> # fraction versus composition and temperature
    >>> # for the phase region in phaseregL
    >>> # based on the equilibrium calculations in eq
    >>> uq.plot_phasereg_prb(eq, phaseregL)
    """

    sns.set_style('ticks')
    prob = get_phase_prob(eq, phaseregL)

    prob = np.squeeze(prob)

    if len(coordplt) != 2:
        print('Error: only 2 coordinates can be passed')

    # format phase region name for plot
    if figname is None:
        figname = ""
        for phase in phaseregL:
            figname += phase + " + "
        figname = figname[:-3]

    plt.figure(num=figname, figsize=figsize)

    if typ == 'grayscale':
        cmap = sns.cm.rocket_r
        ax = sns.heatmap(prob, cmap=cmap)
        ax.invert_yaxis()
    elif typ == 'contour':
        CS = plt.contour(prob, levels=[.05, .5, .95])
        plt.clabel(CS, inline=1, fontsize=10)
    else:
        print('WARNING: invalid plot type selected')

    # define ticks and label for x axis
    ticpts, ticvals = get_ticks(eq, coordplt[0])

    plt.xticks(ticpts, ticvals)
    plt.xticks(fontsize=12)
    plt.xlabel(r'$\mathrm{x_{Mg}}$', fontsize='14')

    # define ticks and label for y axis
    ticpts, ticvals = get_ticks(eq, coordplt[1])
    plt.yticks(ticpts, ticvals)
    plt.ylabel(get_label(coordplt[1]), fontsize='large')
    plt.yticks(fontsize=12)

    # format phase region name for plot
    if title is not None:
        phaseregS = ""
        for phase in phaseregL:
            phaseregS += phase + " + "
        phaseregS = phaseregS[:-3]
        plt.title('Probability of %s' % phaseregS)

    plt.tight_layout()


def plot_superimposed(
        eq, comp, nsp=None, alpha=None,
        phase_label_dict=None,
        xlims=None, cdict=None, figsize=None):

    """
    Plot superimposed binary phase diagrams corresponding
    to multiple parameter sets

    Parameters
    ----------
    eq : xarray object
        Structured equilibirum calculation containing a 'sample'
        dimension correspoinding to different parameter sets
    comp : str
        Label for species to plot on the x-axis,
        e.g. MG for magnesium
    nsp : int, optional
        Number of phase diagrams to superimpose, with the maximum
        given by the number of samples in eq
    alpha : float, optional
        Number between 0 and 1 for the line transparency
    phase_label_dict : dict, optional
        Dictionary with keys given by phase names and corresponding
        strings to use in plotting (e.g. to enable LaTeX labels)
    xlims : list or tuple of float, optional
        List or tuple with two floats corresponding to the
        minimum and maximum molar composition of comp
    cdict : dict, optional
        Dictionary with phase names and corresponding
        colors
    figsize : tuple or list of int or float, optional
        Plot dimensions in inches

    Returns
    -------

    Examples
    --------
    >>> import pickle
    >>> import pduq.uq_plot as uq
    >>> with open('multiple.pkl', 'rb') as buff:
    >>>     eq = pickle.load(buff)
    >>> comp = 'MG'
    >>> # plot the superimposed binary phase diagrams
    >>> # for all of the parameter sets represented by
    >>> # the equilibrium calculations in eq, with
    >>> # the molar fraction of MG on the x-axis
    >>> uq.plot_superimposed(eq, comp)
    """

    phaseL = list(np.unique(eq.get('Phase').values))
    if '' in phaseL:
        phaseL.remove('')
    nph = len(phaseL)

    if nsp is None:
        nsp = len(eq.sample)

    fig, ax = plt.subplots(figsize=figsize)

    # plot the phase boundaries for each parameter set
    for ii in range(nsp):
        plot_binary(
            eq.sel(sample=ii), comp=comp, alpha=alpha, cdict=cdict)
        print('diagram', ii, 'plotted')

    # plot the legend
    nph = len(phaseL)
    colorL = sns.color_palette("cubehelix", nph)
    Tvec = eq.get('T').values
    for ii in range(nph):
        phase = phaseL[ii]
        if cdict is not None:
            color = cdict[phaseL[ii]]
        else:
            color = colorL[ii]
        if phase_label_dict is not None:
            label = phase_label_dict[phase]
        else:
            label = phase
        plt.plot(1.5, Tvec[0], color=color, linestyle='',
                 marker='.', label=label)

    if xlims is None:
        Xvec = eq.get('X_' + comp).values
        Xrng = Xvec.max() - Xvec.min()
        xlims = [Xvec.min() - 0.005*Xrng, Xvec.max() + 0.005*Xrng]

    plt.xlim(xlims)
    plt.ylim([Tvec.min(), Tvec.max()])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(r'$\mathrm{x_{%s}}$' % comp, fontsize='large')
    plt.ylabel('T (K)', fontsize='large')
    plt.legend()
    plt.tight_layout()


def plot_trace(trace, plabelL=None, figsize=None, savefig=False):

    """
    For each parameter in the CALPHAD model, plot all of the MCMC
    chains vs iteration.

    Parameters
    ----------
    trace : array
        Array of parameters with the shape
        [nwalkers, nlinksT, npar], where nwalkers is the number
        of MCMC chains, nlinksT is the number of MCMC iterations,
        and npar is the number of CALPHAD parameters
    plabelL : list, optional
        List of plot labels for the parameters
    figsize : tuple or list of int or float, optional
        Plot dimensions in inches
    savefig : bool, optional
        If savefig is True, plots will be automatically saved

    Returns
    -------

    Examples
    --------
    >>> import numpy as np
    >>> import pduq.uq_plot as uq
    >>> trace = np.loadtxt('trace.csv', delimiter=',')
    >>> plabel = [r"$^{0}G_{CU \\colon MG}^{Laves}$",
    >>>           r"$^{0}L_{CU \\colon MG}^{FCC}$",
    >>>           ...
    >>>           r"$^{0}L_{CU \\colon MG}^{liquid}$"]
    >>> uq.plot_trace(trace, plabelL=plabelL, figsize=[5, 3])
    """

    nwalkers, nlinksT, npar = trace.shape

    for ii in range(npar):

        plt.figure(figsize=figsize)

        # if the number of chains is large enough, plot 2 chains dark
        # and the rest lightly
        if nwalkers > 10:
            for jj in range(nwalkers-2):
                plt.plot(range(nlinksT), trace[jj, :, ii],
                         linestyle='-', marker='', color=[0.3, 0.3, 0.5],
                         lw=.75, alpha=.15)

            for jj in range(nwalkers-2, nwalkers):
                plt.plot(range(nlinksT), trace[jj, :, ii],
                         linestyle='-', marker='', color='k',
                         lw=.75, alpha=.8)
        else:
            for jj in range(nwalkers):
                plt.plot(range(nlinksT), trace[jj, :, ii],
                         linestyle='-', marker='', color=[0.3, 0.3, 0.5],
                         lw=.75, alpha=.6)

        plt.xlabel('iteration number')

        if plabelL is not None:
            plt.ylabel(plabelL[ii], fontsize=15)
        else:
            plt.ylabel('param %s' % ii, fontsize=15)

        plt.xlim([0, nlinksT])
        plt.tight_layout()

        if savefig:
            plt.savefig('chain' + str(ii) + '.png')
