import numpy             as np
import pandas            as pd
import tables            as tb
import matplotlib.pyplot as plt

from scipy               import optimize

import invisible_cities.core.fit_functions as fitf

from mpl_toolkits.mplot3d    import Axes3D

#--- histograming

def select_in_range(vals, range = None):
    if (range is None): range = (np.min(vals), np.max(vals) + 0.1)
    sel = (vals >= range[0]) & (vals < range[1]) 
    return sel

def stats(vals, range = None):
    sel = select_in_range(vals, range)
    vv = vals[sel]
    mean, std, evts, oevts = np.mean(vv), np.std(vv), len(vv), len(vals) - len(vv)
    return evts, mean, std, oevts


def sstats(vals, range = None, formate = '6.2f'):
    evts, mean, std, ovts = stats(vals, range)
    s  = 'events '+str(evts)+'\n'
    s += (('mean {0:'+formate+'}').format(mean))+'\n'
    s += (('std  {0:'+formate+'}').format(std))
    return s


def hist(vv, bins = 100, range = None, stats = True, fig = True, title = '', **kargs):
    if (fig is True): fig = plt.figure()
    if (stats):
        s = sstats(vv, range = range)
        kargs['label'] = s if 'label' not in kargs.keys() else kargs['label'] + '\n' + s
    pt = plt.hist(vv, bins = bins, range = range, histtype='step', **kargs);
    plt.title(title)
    plt.legend();
    return fig, pt

#---- plotting
    
def xyze_qsel(hits, q0 = 0.):
    X, Y, Z, E, Q = hits['X'].values, hits['Y'].values, hits['Z'].values, hits['Ec'].values, hits['Q'].values
    sel = Q >= q0 
    return X[sel], Y[sel], Z[sel], 1000.*E[sel], Q[sel]

    
def plot_track(x, y, z, ene, scale = 10.):

    #fig  = plt.figure(figsize = [8, 3])
    ax   = plt.subplot(111, projection = '3d')
    ax.scatter(x, y, z, c = ene, s = scale * ene, alpha = 0.2, cmap = 'jet')
    ax.set_xlabel('X (mm)');
    ax.set_ylabel('Y (mm)');
    ax.set_zlabel('Z (mm)');
    #ax.colorbar()

    return 


def _proj3d(x, y, z, ene, scale = 10., alpha = 0.1):
    
    fig = plt.gcf()
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(z, x, y, s = scale * ene, c = ene, alpha = alpha, marker='o')
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('x (mm)')
    ax.set_zlabel('y (mm)')
    #ax3D.colorbar()
    return
    
def _blob3d(x, y, z, ene, scale = 400., marker = 'x', alpha = 1.):
    ax = plt.gca()
    ax.scatter(z, x, y, s = scale * ene, c = ene, alpha = alpha, marker = marker)
    return

    
def _proj(u, v, ene, xlabel = '', ylabel = '', scale = 10., alpha = 0.1):
    sc = plt.scatter(u, v, c = ene, s = scale * ene, alpha = alpha, cmap = 'jet', marker = 'o')
    #sc = plt.scatter(u, v, c = ene, s = scale , alpha = alpha, cmap = 'jet', marker = 's')
    ax = plt.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar();
    return ax

def _blob(u, v, ene, scale = 400., marker = 'x', alpha = 1.):
    ax = plt.gca()
    ax.scatter(u, v, c = ene, s = scale * ene, alpha = alpha, marker = marker, cmap = 'jet', lw = 2.)
    ci = plt.Circle((u, v), 21., fill = False);
    ax.add_artist(ci)
    return ax

def graph_event(x, y, z, ene, scale = 10., comment = ''):
    
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 9));
    #plt.subplots(2
    ax3D = fig.add_subplot(221, projection='3d')
    p3d = ax3D.scatter(z, x, y, s = scale * ene, c = ene, alpha = 0.2, marker='o')
    ax3D.set_xlabel('z (mm)')
    ax3D.set_ylabel('x (mm)')
    ax3D.set_zlabel('y (mm)')
    plt.title(comment)
    plt.subplot(2, 2, 2)
    plt.scatter(x, z, c = ene, s = scale * ene, alpha = 0.2, cmap='jet')
    ax = plt.gca()
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    plt.colorbar();
    plt.subplot(2, 2, 3)
    plt.scatter(z, y, c=ene, s=scale*ene, alpha = 0.2, cmap='jet')
    ax = plt.gca()
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('y (mm)')
    plt.colorbar();
    plt.subplot(2, 2, 4)
    plt.scatter(x, y, c=ene, s=scale*ene, alpha=0.2, cmap='jet')
    ax = plt.gca()
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    plt.colorbar();
    plt.tight_layout()
    return

#---- Analysis - more plotting


def inspect_var(df, labels = None, **kargs):
    labels = list(df.columns) if labels is None else labels
    for label in labels:
        hist(df[label], title = label, **kargs)
    return

def hprofile(uvar, vvar, ulabel = '', vlabel = '', urange = None , vrange = None, 
              nbins_profile = 10, fig = True, **kargs):
    fig = plt.figure() if fig is True else plt.gcf()
    urange = urange if urange is not None else (np.min(uvar), np.max(uvar))
    vrange = vrange if vrange is not None else (np.min(vvar), np.max(vvar))
    if 'label' not in kargs.keys(): kargs['label'] = vlabel
    if (nbins_profile):
        xs, ys, eys = fitf.profileX(uvar, vvar, nbins_profile, urange, vrange, std = False)
        plt.errorbar(xs, ys, yerr = eys, **kargs)
    plt.xlabel(ulabel)
    plt.ylabel(vlabel)
    return


def hpscatter(uvar, vvar, ulabel = '', vlabel = '', urange = None , vrange = None, 
              nbins_profile = 10, fig = True, **kargs):
    fig = plt.figure() if fig is True else plt.gcf()
    plt.scatter(uvar, vvar, **kargs)
    kargs['alpha'] = 0.8 
    if ('c' in kargs.keys()): del kargs['c']
    #kargs['c']     = kargs['c'] if 'c' in kargs.keys() else 'black'
    hprofile(uvar, vvar, ulabel, vlabel, urange, vrange, nbins_profile, fig = False, **kargs)
    return
  
def hprofile_zones(uvar, vvar, wvar, nzones = 5, norma = False, bins = 10, formate = '6.2f'):
    enor = np.mean(vvar)/100. if norma is True else 1.
    #print(enor)
    sel_zones, pers = sel_varzones(wvar, nzones);
    for i, isel in enumerate(sel_zones):
        fig = True if i == 0 else False
        ilabel = ('[ {0:' + formate+'}, {1:'+ formate+'}]').format(pers[i], pers[i+1])
        hprofile(uvar[isel], vvar[isel] / enor, nbins_profile = bins, label = ilabel, fig = fig)
    plt.legend(); plt.grid();

def h2profile(x, y, z, nbinsx = 10, nbinsy = 10, xrange = None, yrange = None, zrange = None):
    xc, yc, zp, zpe = fitf.profileXY(x, y, z, nbinsx, nbinsy,
                                     xrange = xrange, yrange = yrange, zrange = zrange, 
                                     std = True, drop_nan = True)
    fig = plt.figure( figsize = (6, 6) )
    zmin, zmax = np.min(z), np.max(z)
    cc  = plt.imshow(zp, vmin = zmin, vmax = zmax, cmap = 'Greys');
    fig = plt.gcf()
    fig.colorbar(cc)
    return
    
def inspect_corr(label, df, labels = None, reverse = False, nbins_profile = 10, **kargs):
    vvar   = df[label].values
    labels = list(df.columns) if labels is None else labels
    for ilabel in labels:
        uvar    = df[ilabel].values
        if (reverse is False):
            hpscatter(uvar, vvar, ilabel,  label, nbins_profile = nbins_profile, **kargs)
        else:
            hpscatter(vvar, uvar,  label, ilabel, nbins_profile = nbins_profile, **kargs)  
        plt.grid()
    return


def inspect_hprof(label, df, labels = None, reverse = False, nbins_profile = 20, **kargs):
    vvar   = df[label].values
    labels = list(df.columns) if labels is None else labels
    for ilabel in labels:
        uvar    = df[ilabel].values
        if (reverse is False):
            hprofile(uvar, vvar, ilabel,  label, nbins_profile = nbins_profile, **kargs)
        else:
            hprofile(vvar, uvar,  label, ilabel, nbins_profile = nbins_profile, **kargs)
        plt.grid()
    return

def plot_corrmatrix(xdf, xlabels):
    _df  = xdf[xlabels]
    corr = _df.corr()
    fig = plt.figure(figsize=(12, 10))
    #corr.style.background_gradient(cmap='Greys').set_precision(2)
    plt.matshow(abs(corr), fignum = fig.number, cmap = 'Greys')
    plt.xticks(range(_df.shape[1]), _df.columns, fontsize=14, rotation=45)
    plt.yticks(range(_df.shape[1]), _df.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    return

def plot_energy_resolution(ene, bins = 100, range = None, fig = True, label = '', formate = '6.2f'):
    range = range if range is not None else (np.min(ene), np.max(ene))
    pval, pcov = hgaussfit(ene, bins, range)
    xs  = np.linspace(*range, bins);
    ss  = label + '\n'
    ss += (('mean   {0:' + formate + '}').format(pval[1]))+'\n'
    ss += (('sigma  {0:' + formate + '}').format(pval[2]))
    plt.hist(ene, bins, range, density = True, label = ss, histtype = 'step');
    plt.plot(xs, fgauss(xs, *pval));
    plt.legend();
    fscale = 1. # np.sqrt(1.66/2.48)
    print( ('energy resolution = {0:' + formate + '} FWHM').format(235.*fscale*pval[2]/pval[1]))
    return

def plot_energy_vs_dz(ene, dz, bins = 20, erange = None, dzrange = None, fig = True, formate = '6.2f'):
    erange  = erange  if erange  is not None else (np.min(ene), np.max(ene))
    dzrange = dzrange if dzrange is not None else (np.min(dz) , np.max(dz))
    hprofile(dz, ene, urange = dzrange, vrange = erange, nbins_profile = bins, fig = fig);
    plt.grid();
    pars, errs = hlinefit(dz, ene, bins, dzrange, erange); 
    xs = np.linspace(*dzrange, bins);
    ys = [fline(xi, *pars) for xi in xs]
    plt.plot(xs, ys);
    ss  = (('a  {0:' + formate + '}').format(pars[0]))+'\n'
    ss += (('b  {0:' + formate + '}').format(pars[1]))
    plt.text(dzrange[0], np.mean(ene), ss) #, bbox = dict(facecolor = 'white', alpha = 0.1))
    return pars, errs


def hist_sample(df, sels, labels = None, bins = 60, range = None, title = '', **kargs):
    labels = list(df.columns) if labels is None else labels
    for label in labels:
        for i, isel in enumerate(sels):
            density = True if i == 0 else False
            uvar = df[label][isel]
            hist(uvar, bins = bins, range = range, density = True, fig = density, **kargs)
        plt.xlabel(label);
        plt.title(title);

        
#-- Analysis fitting again

def fgauss(x, a, b, c):
    return a*np.exp(-(x-b)**2/(2*c**2))

def hgaussfit(x, bins = 100, range = None):
    range = range if range is not None else (np.min(x), np.max(x))
    yc, xe = np.histogram(x, bins, range, density = True)
    xc = 0.5*(xe[1:] + xe[:-1])
    p0s = (1., np.mean(x), np.std(x))
    fpar, fcov = optimize.curve_fit(fgauss, xc, yc, p0s)
    return fpar, np.sqrt(np.diag(fcov))

def fline(x, a, b):
    return a*x + b

def hlinefit(x, y, bins = 20, xrange = None, yrange = None):
    xrange = xrange if xrange is not None else (np.min(x), np.max(x))
    yrange = yrange if yrange is not None else (np.min(y), np.max(y))
    xs, ys, eys = fitf.profileX(x, y, bins, xrange, yrange, std = False)
    a0, b0      = (np.max(ys) - np.min(ys))/(np.max(xs) - np.min(xs)), np.min(ys)
    fpar, fcov = optimize.curve_fit(fline, xs, ys, (a0, b0))
    return fpar, np.sqrt(np.diag(fcov))

def plt_hlinefit(x, y, bins = 20, xrange = None, yrange = None, fig = True, formate = '6.2f'):
    yrange = yrange if yrange is not None else (np.min(y), np.max(y))
    xrange = xrange if xrange is not None else (np.min(x), np.max(x))
    hprofile(x, y, urange = xrange, vrange = yrange, nbins_profile = bins, fig = fig);
    plt.grid();
    pars, errs = hlinefit(x, y, bins, xrange, yrange); 
    xs = np.linspace(*xrange, bins);
    ys = [fline(xi, *pars) for xi in xs]
    plt.plot(xs, ys);
    ss  = (('a  {0:' + formate + '}').format(pars[0]))+'\n'
    ss += (('b  {0:' + formate + '}').format(pars[1]))
    plt.text(xrange[0], np.mean(y), ss) #, bbox = dict(facecolor = 'white', alpha = 0.1))
    return pars, errs
