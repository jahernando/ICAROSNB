import numpy             as np
import pandas            as pd
import tables            as tb
import matplotlib.pyplot as plt

from scipy               import optimize

to_df = pd.DataFrame.from_records

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


def esmeralda_event_display(evt, dftracks, dfhits, q0 = 0., scale = 10.):
    cct    = dftracks.get_group(evt)
    labels = list(cct.columns)
    for label in labels:
        print(label, ' : ', cct[label].values)
    #print(cct)
    cch = dfhits  .get_group(evt)

    x, y, z, ene, q = xyze_qsel(cch, q0 = 0.)
    xb1, yb1, zb1, eb1 = [cct[label].values for label in ('blob1_x', 'blob1_y', 'blob1_z', 'eblob1')]
    xb2, yb2, zb2, eb2 = [cct[label].values for label in ('blob2_x', 'blob2_y', 'blob2_z', 'eblob2')]
    
    xe1, ye1, ze1 = [cct[label].values for label in ('extreme1_x', 'extreme1_y', 'extreme1_z')]
    xe2, ye2, ze2 = [cct[label].values for label in ('extreme2_x', 'extreme2_y', 'extreme2_z')]
    mk1, mk2 = ('<', '>') if ze1 < ze2 else ('>', '<')
    mk1, mk2 = '|', '|'
    
    
    fig = plt.figure( figsize = (12, 9) )

    plt.subplot(2, 2, 1)
    _proj3d(x, y, z, ene)
    _blob3d(xb1, yb1, zb1, eb1); _blob3d(xb2, yb2, zb2, eb2)
    _blob3d(xe1, ye1, ze1, eb1, marker = '|'); _blob3d(xe2, ye2, ze2, eb2, marker = '|')
    
    
    plt.subplot(2, 2, 2)
    _proj(x, z, ene, 'x (mm)', ' z (mm)') 
    _blob(xb1, zb1, eb1)              ; _blob(xb2, zb2, eb2)
    #_blob(xe1, ze1, eb1, marker = mk1); _blob(xe2, ze2, eb2, marker = mk2)


    
    plt.subplot(2, 2, 3)
    _proj(z, y, ene, 'z (mm)', ' y (mm)') 
    _blob(zb1, yb1, eb1)              ; _blob(zb2, yb2, eb2)
    #_blob(ze1, ye1, eb1, marker = mk1); _blob(ze2, ye2, eb2, marker = mk2)

    
    plt.subplot(2, 2, 4)
    _proj(x, y, ene, 'x (mm)', ' y (mm)') 
    _blob(xb1, yb1, eb1)              ; _blob(xb2, yb2, eb2)
    #_blob(xe1, ye1, eb1, marker = mk1); _blob(xe2, ye2, eb2, marker = mk2)

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

"""
def gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def fit_gauss_hist(x, y):
    mean = np.sum(x * y)
    sigma = sum(y * (x - mean)**2)
    #do the fit!
    popt, pcov = optimize.curve_fit(gauss, x, y, p0 = [1, mean, sigma])
    return popt, pcov
    
def fit_gauss(x, bins = 100, range = None):
    sel = select_in_range(x, range)
    xi = x[sel]
    yc, xed = np.histogram(xi, bins)
    xc = 0.5*(xed[1:] + xed[:-1])
    popt, pcov = fit_gauss_hist(xc, yc)
    yf = gauss(xc, *popt)
    return popt, pcov, xc, yf
"""

#--- selections

def sel_updown(sdf):
    sela = (sdf.y_min > 0.)
    selb = (sdf.y_max < 0.)
    return (sela, selb)

def sel_rightleft(sdf):
    sela = (sdf.x_min > 0.)
    selb = (sdf.x_max < 0.)
    return (sela, selb)

def sel_corona(sdf):
    sela = (sdf['r_max'] <= 100.)
    selb = (sdf['r_max'] >  100.)
    return (sela, selb)

def sel_zzones(sdf):
    sela = (sdf.z_min > 0.)   & (sdf.z_max < 200.)
    selb = (sdf.z_min > 200.) & (sdf.z_max < 400.)
    selc = (sdf.z_min > 400.) & (sdf.z_max < 600.)
    return (sela, selb, selc)


def sel_varzones(var, percentiles = 4):
    if (type(percentiles) is int):
        percentiles = np.linspace(0., 100., percentiles + 1)
    sels = []
    pers = [np.percentile(var, pi) for pi in percentiles]
    for i in range(len(percentiles)-1):
        isel   = (var >= pers[i])  & (var < pers[i+1])
        sels.append(isel)
    return sels, pers
                     

def sel_dzzones(sdf):
    z0 = np.percentile(sdf.dz, 33.)
    z1 = np.percentile(sdf.dz, 66.)
    sela = (sdf.dz <= z0)
    selb = (sdf.dz > z0)  & (sdf.dz <= z1)
    selc = (sdf.dz > z1)
    return (sela, selb, selc)

def sel_dzsmalllength(sdf):
    ksel = sdf.dz < 45.
    z0 = np.percentile(sdf[ksel].length, 33.)
    z1 = np.percentile(sdf[ksel].length, 66.)
    sela = (sdf.dz < 45.) & (sdf.length <= z0)
    selb = (sdf.dz < 45.) & (sdf.length >  z0) & (sdf.length <= z1)
    selc = (sdf.dz < 45.) & (sdf.length >  z1) 
    return (sela, selb, selc)
                      

def sel_b1zones(sdf):
    z0 = np.percentile(sdf.dzb, 33.)
    z1 = np.percentile(sdf.dzb, 66.)
    sela = (sdf.dzb <= z0)
    selb = (sdf.dzb > z0)  & (sdf.dzb <= z1)
    selc = (sdf.dzb > z1)
    return (sela, selb, selc)

def sel_fb1zones(sdf):
    z0 = np.percentile(sdf.fzb, 33.)
    z1 = np.percentile(sdf.fzb, 66.)
    sela = (sdf.fzb <= z0)
    selb = (sdf.fzb > z0)  & (sdf.fzb <= z1)
    selc = (sdf.fzb > z1)
    return (sela, selb, selc)

def sel_eifzones(sdf):
    z0 = np.percentile(sdf.eif, 33.)
    z1 = np.percentile(sdf.eif, 66.)
    sela = (sdf.eif <= z0)
    selb = (sdf.eif > z0)  & (sdf.eif <= z1)
    selc = (sdf.eif > z1)
    return (sela, selb, selc)

def sel_extreme_two_blobs(sdf):

    sela = (sdf.fdzb2 > 0.8) & (sdf.fdzb1 < 0.2) & (sdf.eblob2 > 0.160)
    selb = (sdf.fdzb2 < 0.2) & (sdf.fdzb1 > 0.8) & (sdf.eblob2 > 0.160)
    return (sela, selb)
    
#--- Analysis slices

def hprofile_slices(sdf, nslices, label = 'slce', bins = 10, factor = 1000., norma = False):
    norma = 1. if norma is False else sdf.dz/(1.*nslices)
    for i in range(nslices):
        fig = True if i == 0 else False
        hprofile(sdf.dz, factor * sdf[label+str(i)]/ norma, 
                 vlabel = label+str(i), fig = fig, nbins_profile = bins)
    plt.xlabel('dz (mm)');
    plt.ylabel('E (keV)')
    plt.grid();
    plt.legend();
    return

def dfslices(xdf, nslices):
    
    for i in range(nslices):
        
        slck  = i * np.ones(len(xdf))
        
        slcz   =  xdf['slcz' +str(i)].values
        slcdz  =  xdf['slcz' +str(i)].values - xdf['z_min'].values
        slcdzr =  xdf['slcz' +str(i)].values - xdf['z_max'].values
        slcdz1 =  xdf['slcz' +str(i)].values - xdf['blob1_z'].values
        slcdz2 =  xdf['slcz' +str(i)].values - xdf['blob2_z'].values

        slce  = nslices * xdf['slce' +str(i)].values / (xdf.dz.values)
        slcec = nslices * xdf['slcec'+str(i)].values / (xdf.dz.values)
        slcq  = nslices * xdf['slcq' +str(i)].values / (xdf.dz.values)

        dzs   = np.copy(slcdz)  if i == 0 else np.concatenate((dzs , slcdz))
        dzrs  = np.copy(slcdzr) if i == 0 else np.concatenate((dzrs, slcdzr))
        dz1s  = np.copy(slcdz1) if i == 0 else np.concatenate((dz1s, slcdz1))
        dz2s  = np.copy(slcdz2) if i == 0 else np.concatenate((dz2s, slcdz2))

        ks    = np.copy(slck)  if i == 0 else np.concatenate((ks  , slck)) 
        zs    = np.copy(slcz)  if i == 0 else np.concatenate((zs  , slcz))
        ees   = np.copy(slce)  if i == 0 else np.concatenate((ees , slce))
        eecs  = np.copy(slcec) if i == 0 else np.concatenate((eecs, slcec))            
        qqs   = np.copy(slcq)  if i == 0 else np.concatenate((qqs , slcq))

    data = {'i': ks, 'ec': ees, 'ecc': eecs , 'q': qqs   , 'z': zs, 
            'dz': dzs, 'dzr': dzrs , 'dz1': dz1s, 'dz2': dz2s}
    return pd.DataFrame(data)

def slices_total(sdf, slices, label = 'slce'):
    
    if (type(slices) is int): slices = range(slices)
    vals = [sdf[label +str(i)] for i in slices]
    for i in range(len(slices)):
        etot = np.copy(vals[0]) if i == 0 else etot + vals[i]
    return etot


def plt_slices_profile(ldf, rdf = None, llabel = '', rlabel = '', 
                    dzrange = None, erange = None, bins = 40, varlabel = 'ec', factor = 1000.):

    hprofile(ldf.dz, factor * ldf[varlabel] , urange = dzrange, vrange = erange, 
                nbins_profile = bins, label = llabel);
    if (rdf is not None):
        hprofile(- rdf.dzr, factor *rdf[varlabel] , urange = dzrange, vrange = erange, 
                    nbins_profile = bins, label = llabel, fig = False);
    plt.grid(); #plt.ylim(0., 50.)
    plt.xlabel('dz (mm)'); plt.ylabel('dEdz keV/mm')
    if (llabel != ''): plt.legend()


    hprofile(ldf.dz1, factor * ldf[varlabel] , urange = dzrange, vrange = erange, 
                nbins_profile = bins, label = llabel);
    if (rdf is not None):
        hprofile(-rdf.dz1, factor *rdf[varlabel] , urange = dzrange, vrange = erange, 
                    nbins_profile = bins, label = rlabel, fig = False);
    plt.grid(); #plt.ylim(0., 50.)
    plt.xlabel('dz to bl (mm)'); plt.ylabel('dEdz keV/mm')
    if (llabel != ''): plt.legend()    

    return


def plt_slices(sdf, nslices, varlabel = 'e', factor = 1000., norma = False, fig = True, **kargs):
    enor = np.sqrt(len(sdf)-1)
    norma = np.ones(len(sdf)) if norma is False else sdf.dz/(1.*nslices)
    ys  = [factor * np.mean(sdf['slc'+varlabel+str(i)].values/norma)      for i in range(nslices)]
    eys = [factor * np.std (sdf['slc'+varlabel+str(i)].values/norma)/enor for i in range(nslices)]
    xs  = range(nslices)
    fig = plt.figure() if fig else plt.gcf()
    plt.errorbar(xs, ys, yerr = eys, **kargs)
    return


#--- Analysis

def complete_df_from_file(filename, type_peak = 'dsp'):
 
    print(filename)
    f = tb.open_file(filename, 'r')
    
    dft = to_df(f.root.PAOLINA.Tracks.read())
    dfe = to_df(f.root.PAOLINA.Events.read())
    dfs = to_df(f.root.PAOLINA.Summary.read())
    
    idf, idfs, idfe = complete_df(dft, dfs, dfe, type_peak = type_peak)
    return idf, idfs, idfe

def complete_df_from_files(filenames, type_peak = 'dsp'):
    
    if (type(filenames) is not list):
        return complete_df_from_file(filenames)
    
    for i, filename in enumerate(filenames):
        idf, _, _ = complete_df_from_file(filename, type_peak)
        df  = idf if i == 0 else df.append(idf, ignore_index = True)
    return df
    
def complete_df(df, dfr, dfe, type_peak = 'dsp'):
    # df  - tracks
    # dfr - summary
    # dfe - events 
    
    def transfer(df, dfr, label):
        var = np.copy(df.energy.values)
        cc = df.groupby('event')
        for ievt, icc in cc:
            isel = dfr.event == ievt
            jsel = df .event == ievt
            var[jsel] = (dfr[label][isel].values)[0]
        df[label] = var
        return df

    print('transfering variables...')
    df = transfer(df, dfr, 'time')
    df = transfer(df, dfr, 'S2e0')
    df = transfer(df, dfr, 'S2q0')
    df = transfer(df, dfr, 'S1e')
    df = transfer(df, dfr, 'ntrks')
    df = transfer(df, dfr, 'nS2')
    
    print('computing variables...')
    
    df['qe0'] = df['S2q0'].values  /df['S2e0'].values
    
    df['fec'] = (1.e6*df['energy'].values)/df['S2e0'].values

    dz = df['z_max'].values - df['z_min'].values
    df['dz'] = dz 
    
    dx = df['x_max'].values - df['x_min'].values
    df['dx'] = dx 
    
    dy = df['y_max'].values - df['y_min'].values
    df['dy'] = dy 
    
    zmed = 0.5*(df['z_max'].values + df['z_min'].values)
    df['zmed'] = zmed
    
    ecln = df['energy'].values/(1 - 2.76e-4*dz)
    df['ecln'] = ecln
    
    dzb = df['blob1_z'].values - df['blob2_z'].values
    df['dzb']  = dzb
    df['adzb'] = abs(dzb)
    
    dzb1 = df['blob1_z'].values - df['z_min'].values
    df['dzb1'] = dzb1
    
    df['fdzb1'] = dzb1/dz
    
    dzb2 = df['blob2_z'].values - df['z_min'].values
    df['dzb2'] = dzb2
    
    df['fdzb2'] = dzb2/dz
   
    brat = df['eblob2'].values / df['eblob1'].values
    df['brat'] = brat
    
    et1 = df['energy'].values - df['eblob1'].values 
    df['et1'] = et1
    
    et2 = df['energy'].values - df['eblob1'].values - df['eblob2'].values
    df['et2'] = et2
    
    fzb = df['dzb'].values/df['dz'].values
    df['fzb'] = fzb
    
    # ORIGINAL a1, b1, dz1, a2 = (0.70, 1681., 60., 0.38) if type_peak == 'dsp' else (0.94, 2784., 90., 0.38)
    a1, b1, dz1, a2 = (0.70, 1681., 60., -0.32) if type_peak == 'dsp' else (0.94, 2784., 90., -0.52)
    # best phpfit
    #a1, b1, dz1, a2 = 0.70, 1681., 60., 0.38 # best dspfit
    alpha1, alpha2 = 2.*a1/b1, 2.*a2/b1
    print('correction factors ', alpha1, alpha2, dz1)
    
    def _cene(ene, dz, a1 = a1, b1 = b1 , dz1 = dz1, a2 = a2, scale = 1.):
        a2 = a2 if a2 is not None else a1
        delta = a1 * dz / b1
        slong = dz > dz1
        #delta[slong] = a1 * dz1 / b1 + a2 * (dz[slong] - dz1)/ b1
        delta[slong] = delta[slong] + a2 * (dz[slong] - dz1)/ b1
        return ene * np.exp(delta * scale)
    
    alp1, alp2, dzalp1 = 0.70e-3, -0.20e-3, 90.
    def _cene_exp(ene, dz, alpha1 = alp1, alpha2 = alp2, dzlim = dzalp1, scale = 1.):
        delta = alpha1 * dz 
        slong = dz > dzlim
        delta[slong] += alpha2 * (dz[slong] - dzlim)
        return ene * np.exp(delta * scale)
    
    ene = df.energy.values
    q0  = df.S2q0.values
    dz  = df.dz.values
    df['ecdz'] = _cene(ene, dz)
    df['qcdz'] = _cene(q0 , dz, scale = 2.6)
    
    print('corrections...')
    
    cc    = dfe.groupby('event')
    nevts = len(set(df.event))
    ei    = np.zeros(nevts)
    er    = np.zeros(nevts)
    for ievt, icc in cc:
        isel  = df.event == ievt
        
        zmed  = 0.5*(np.min(icc.Z) + np.max(icc.Z))
        ei[isel] = np.sum(icc.Ec[icc.Z <= zmed])
        er[isel] = np.sum(icc.Ec[icc.Z >  zmed])    
    df['ei'] = ei
    df['er'] = er
    df['ee'] = ei + er
    df['eif'] = ei / (er + ei)

    dz = np.copy(dfe.Z.values)
    cc = dfe.groupby('event')
    for ievt, icc in cc:
        sel_evt      = dfe.event == ievt
        dz[sel_evt] -= float(np.min(icc['Z'].values))  
    dfe['dz'] = dz
    
    
    # Slices
    nslices = 5
    icenter = int(nslices/2)
    esis  = [np.zeros(nevts) for i in range(nslices)]
    ecsis = [np.zeros(nevts) for i in range(nslices)]
    qsis  = [np.zeros(nevts) for i in range(nslices)]
    zsis  = [np.zeros(nevts) for i in range(nslices)]
    
    def _jsel(icc, zss, i):
        if (i < icenter):
            jsel = (icc.Z >= zss[i]) & (icc.Z < zss[i+1])
        elif (i == icenter):
            jsel = (icc.Z >= zss[i]) & (icc.Z <= zss[i+1])
        elif (i > icenter):
            jsel = (icc.Z >  zss[i]) & (icc.Z <= zss[i+1])
        return jsel
        
    for ievt, icc in cc:
        isel = df.event == ievt
        zss  = np.linspace(np.min(icc.Z), np.max(icc.Z), nslices +1)
        zmin = np.min(icc.Z)
        for i in range(nslices):
            jsel           = _jsel(icc, zss, i)
            esis[i][isel]  = np.sum( icc.Ec[jsel])
            zsis[i][isel]  = np.mean(icc.Ec[jsel] * icc.Z [jsel])/np.mean(icc.Ec[jsel])
            qsis[i][isel]  = np.sum( icc.Q [jsel])
            ecsis[i][isel] = np.sum(_cene(icc.Ec[jsel], icc.Z[jsel] - zmin, scale = 2. ))
    for i in range(nslices):
        df['slce' + str(i)] = esis[i]
        df['slcec'+ str(i)] = ecsis[i]
        df['slcz' + str(i)] = zsis[i]
        df['slcq' + str(i)] = qsis[i]
                       
    # slices of the blob
    zblob = 15.
    slcb1e = np.zeros(nevts)
    slcb1z = np.zeros(nevts)
    slcb2e = np.zeros(nevts)
    slcb2z = np.zeros(nevts)
    for ievt, icc in cc:
        isel = df.event == ievt
        zb1  = float(df.blob1_z[isel])
        zb2  = float(df.blob2_z[isel])
        jselb1 = (icc.Z >= zb1 - zblob) & (icc.Z <= zb1 + zblob)  
        jselb2 = (icc.Z >= zb2 - zblob) & (icc.Z <= zb2 + zblob) 
        slcb1e[isel] = np.sum( icc.Ec[jselb1])
        slcb1z[isel] = np.mean(icc.Ec[jselb1] * icc.Z[jselb1])/np.mean(icc.Ec[jselb1])
        slcb2e[isel] = np.sum( icc.Ec[jselb2])
        slcb2z[isel] = np.mean(icc.Ec[jselb2] * icc.Z[jselb2])/np.mean(icc.Ec[jselb2])
    df['slcb1e'] = slcb1e
    df['slcb1z'] = slcb1z
    df['slcb2e'] = slcb2e
    df['slcb2z'] = slcb2z

    
    # before and after the blob
    z0, z1 = 10., 30.
    b1l  = np.zeros(nevts)
    b1r  = np.zeros(nevts)
    for ievt, icc in cc:
        isel = df.event == ievt
        zb   = float(df.blob1_z[isel])
        jsel = (icc.Z < zb - z0) #& (icc.Z < zb - z0)
        ksel = (icc.Z > zb + z0) #& (icc.Z < zb + z1)
        b1l[isel]  = np.sum(icc.Ec[jsel])
        b1r[isel]  = np.sum(icc.Ec[ksel])
    df['b1l'] = b1l
    df['b1r'] = b1r                          
                             
    
    cc  = dfe.groupby('event')
    nevts = len(set(df.event))
    ecc = np.zeros(nevts)
    qcc = np.zeros(nevts)

    # exponential correction
    #----
    #alpha1, alpha2 = 0.75e-3, 0.55e-3 #solution for php
    #alpha1, alpha2  = 0.70e-3, 0.38e-3 #solution for dsp
    #alpha1, alpha2 = 0.90e-3, 0.38e-3 #solution for php
    #al, b1, dz1, al2  = 0.45e-3, 0.45e-3 # best php
    #alpha1, alpha2  = 0.70e-3, 0.50e-3  # best flat php dz

    #alpha1, alpha2, dzmax = 2.*a1/b1, 2.*a2/b1, dz1
    #def _fune(e, dz, scale = 1.):
    #    alphas = alpha1 * dz
    #    slong = np.where(dz > dzmax)
    #     alphas[slong] = alpha1 * dzmax  + alpha2 * (dz[slong] - dzmax) 
    #    return 1. * e * np.exp(alphas * scale)

    #def funq(q, dz):
    #    idz = np.copy(dz)
    #    alphas = alpha1 * np.ones(len(idz))
    #    alphas[np.where(idz >= dzmax)] = alpha2
    #    idz[np.where(idz >= dzmax)] = dzmax
    #    return 1.     * q * np.exp(11 * alphas * idz)
    
    for ievt, icc in cc:
        isel = df.event == ievt
        #iecc = np.sum(_fune(icc.Ec, icc.dz))
        #iqcc = np.sum(_fune(icc.Q , icc.dz, scale = 10.))
        iecc = np.sum(_cene(icc.Ec, icc.dz, scale = 2.))
        iqcc = np.sum(_cene(icc.Q , icc.dz, scale = 20.))
        ecc[isel] = iecc 
        qcc[isel] = iqcc
    df['ecc'] = ecc
    df['qcc'] = qcc
    
    
    return df, dfr, dfe