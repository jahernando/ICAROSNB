import time

import numpy             as np
import scipy             as sc
import scipy.stats       as st
import tables            as tb

from typing    import Callable
from typing    import Tuple
from typing    import List

import invisible_cities.core    .system_of_units_c as system_of_units
import invisible_cities.core    .fit_functions     as fitf
import invisible_cities.database.load_db           as db

import myhistos     as ht
import voxels       as vx

import matplotlib.pyplot as plt

#
# Configuration
#-------------------

units = system_of_units.SystemOfUnits()

class Singleton(object):

    def __new__(cls, *args, **kargs):

        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kargs)
        return cls._instance


# class WF:
#
#      def __init__(self, buffer_size, bin_size, nsensors):
#          self.buffer_size  = buffer_size
#          self.bin_size     = bin_size
#          self.nsensors     = nsensors
#          self.nbins        = int(buffer // time_bin)
#          self.bins         = np.arange(self.nbins) *  bin_size
#          self.nsensors     = nsensors
#          self._wf           = np.zeros(self.nbins, nsensors)
#
#      def wf(self):
#          return np.copy(self.wf)


class DetSimParameters(Singleton):

    def __init__(self):

        self.ws                     = 60.0 * units.eV
        self.wi                     = 22.4 * units.eV
        self.fano_factor            = 0.15
        self.conde_policarpo_factor = 0.10

        self.drift_velocity         = 1.00 * units.mm / units.mus
        self.lifetime               = 8.e5 * units.mus

        self.transverse_diffusion   = 1.00 * units.mm / units.cm**0.5
        self.longitudinal_diffusion = 0.30 * units.mm / units.cm**0.5
        self.voxel_sizes            = 2.00 * units.mm, 2.00 * units.mm, 0.2 * units.mm

        self.el_gain                = 1e3
        self.el_gain_sigma          = np.sqrt(self.el_gain * self.conde_policarpo_factor)

        self.EP_z                   = 632 * units.mm
        self.EL_dz                  =  5  * units.mm
        self.TP_z                   = -self.EL_dz
        self.EL_dtime               =  self.EL_dz / self.drift_velocity
        self.EL_pmt_time_sample     = 200 * units.ns
        self.EL_sipm_time_sample    =   1 * units.mus


        self.wf_buffer_time         = 1000 * units.mus
        self.wf_pmt_bin_time        =   25 * units.ns
        self.wf_sipm_bin_time       =    1 * units.mus

        self.npmts                  = 12
        self.nsipms_side            = 22
        self.sipm_pitch             =  10  * units.mm

        self.dummy_detector()

    def dummy_detector(self):

        self.x_pmts   = np.zeros(self.npmts)
        self.y_pmts   = np.zeros(self.npmts)

        indices       = np.arange(-self.nsipms_side, self.nsipms_side + 1)
        self.sipms    = [(self.sipm_pitch * i, self.sipm_pitch * j) for i in indices for j in indices]
        self.nsipms   = len(self.sipms)

        self.nsipms   = len(self.sipms)

        self.x_sipms  = np.array([sipm[0] for sipm in self.sipms])
        self.y_sipms  = np.array([sipm[1] for sipm in self.sipms])

        self.xybins   = vx.bins_edges(np.sort(self.x_sipms), 0.5 * self.sipm_pitch), \
                        vx.bins_edges(np.sort(self.y_sipms), 0.5 * self.sipm_pitch)


    def configure(self, **kargs):
        for key, val in kargs.items():
            setattr(obj, key, val)

#        self.xybins   = vx.bins_edges(np.sort(detsimparams.x_sipms), 0.5 * sipm_pitch), vx.bins_edges(np.sort(detsimparams.y_sipms), 0.5 * sipm_pitch)

detsimparams = DetSimParameters()

#
# Utilities
#----------------

def _array(xs, ns):
    return np.concatenate([np.repeat(xi, ni) for xi, ni in zip(xs, ns)])

#
#  Secondary electrons
#------------------------

def generate_deposits(x0 = 0., y0 = 0., z0 = 300. * units.mm,
                      xsigma = 1.00 * units.mm, size = 1853, dx = 1.00 * units.mm):

    xs = np.random.normal(x0, xsigma, size)
    ys = np.random.normal(y0, xsigma, size)
    zs = np.random.normal(z0, xsigma, size)

    xbins = vx.equal_size_bins(np.min(xs), np.max(xs), dx)
    ybins = vx.equal_size_bins(np.min(ys), np.max(ys), dx)
    zbins = vx.equal_size_bins(np.min(zs), np.max(zs), dx)

    vxs, vys, vzs, vnes = vx.voxels3d(xs, ys, zs, xbins, ybins, zbins)
    vnes = detsimparams.wi * vnes
    return (vxs, vys, vzs, vnes)


def generate_electrons(energies    : np.array,
                       wi          : float = detsimparams.wi,
                       fano_factor : float = detsimparams.fano_factor) -> np.array:
    nes  = np.array(energies/wi, dtype = int)
    pois = nes < 10
    nes[ pois] = np.random.poisson(nes[pois])
    nes[~pois] = np.round(np.random.normal(nes[~pois], np.sqrt(nes[~pois] * fano_factor)))
    return nes


def drift_electrons(zs             : np.array,
                    electrons      : np.array,
                    lifetime       : float = detsimparams.lifetime,
                    drift_velocity : float = detsimparams.drift_velocity) -> np.array:
    ts  = zs / drift_velocity
    nes = np.copy(electrons -np.random.poisson(electrons * (1. - np.exp(-ts/lifetime))))
    nes[nes < 0] = 0
    return nes


def diffuse_electrons(xs                     : np.array,
                      ys                     : np.array,
                      zs                     : np.array,
                      electrons              : np.array,
                      transverse_diffusion   : float = detsimparams.transverse_diffusion,
                      longitudinal_diffusion : float = detsimparams.longitudinal_diffusion,
                      voxel_sizes            : Tuple = detsimparams.voxel_sizes) -> Tuple:
    nes = electrons
    xs = _array(xs, nes); ys = _array(ys, nes); zs = _array(zs, nes)
    #zzs = np.concatenate([np.array((zi,)*ni) for zi, ni in zip(zs, ns)])
    sqrtz = zs ** 0.5
    vxs  = np.random.normal(xs, sqrtz * transverse_diffusion)
    vys  = np.random.normal(ys, sqrtz * transverse_diffusion)
    vzs  = np.random.normal(zs, sqrtz * longitudinal_diffusion)
    vnes = np.ones(vxs.size)
    dx, dy, dz = voxel_sizes
    xbins = vx.equal_size_bins(np.min(vxs), np.max(vxs), dx)
    ybins = vx.equal_size_bins(np.min(vys), np.max(vys), dy)
    zbins = vx.equal_size_bins(np.min(vzs), np.max(vzs), dz)
    vxs, vys, vzs, vnes = vx.voxels3d(vxs, vys, vzs, xbins, ybins, zbins)
    return (vxs, vys, vzs, vnes)

#
#  EL
#-----------

def psf(dx, dy, dz, factor = 1.):
    return factor * np.abs(dz) / (2 * np.pi) / (dx**2 + dy**2 + dz**2)**1.5

psf_pmt  = lambda dx, dy : psf(dx, dy, detsimparams.EP_z, factor = 25e3)

psf_sipm = lambda dx, dy : psf(dx, dy, detsimparams.TP_z, factor = 10.)

def generate_EL_photons(electrons      : np.array,
                        el_gain        : float = detsimparams.el_gain,
                        el_gain_sigma  : float = detsimparams.el_gain_sigma):
    nphs      = electrons * np.random.normal(el_gain, el_gain_sigma, size = electrons.size)
    return nphs

def estimate_pes_at_sensors(xs        : np.array,
                            ys        : np.array,
                            x_sensors : np.array,
                            y_sensors : np.array,
                            photons   : np.array,
                            psf       : Callable) -> np.array:
    dxs     = xs[:, np.newaxis] - x_sensors
    dys     = xs[:, np.newaxis] - y_sensors
    photons = photons[:, np.newaxis] + np.zeros_like(x_sensors)

    pes = photons * psf(dxs, dys)
    #pes = np.random.poisson(pes)
    return pes

def estimate_pes_at_pmts(xs      : np.array,
                         ys      : np.array,
                         photons : np.array) -> np.array:
    return estimate_pes_at_sensors(xs, ys, detsimparams.x_pmts, detsimparams.y_pmts, photons, psf_pmt)


def estimate_pes_at_sipms(xs      : np.array,
                          ys      : np.array,
                          photons : np.array) -> np.array:
    return estimate_pes_at_sensors(xs, ys, detsimparams.x_sipms, detsimparams.y_sipms, photons, psf_sipm)

#
#  WFs
#-----------

def create_wfs(ts             : np.array, # nevents
               pes            : np.array, # nevents x nsensors
               wf_buffer_time : float = detsimparams.wf_buffer_time,
               wf_bin_time    : float = detsimparams.wf_pmt_bin_time,
               el_time_sample : float = detsimparams.EL_pmt_time_sample,
               el_time        : float = detsimparams.EL_dtime):

    nsize = int(el_time        // el_time_sample)
    nbins = int(wf_buffer_time // wf_bin_time )

    wftimes = np.arange(nbins + 1) * wf_bin_time
    dts     = el_time_sample * np.arange(nsize)

    def _times(its, ipes):
        nts  = its[:, np.newaxis] + dts
        nns   = np.random.poisson(ipes / nsize, size = (nsize, ipes.size))
        return np.repeat(nts.flatten(), nns.T.flatten())

    def _wf(itimes):
        idtimes     = (itimes // wf_bin_time).astype(int)
        ids, counts = np.unique(idtimes, return_counts=True)
        iwf         = np.zeros(nbins + 1)
        iwf[ids]    = counts
        return iwf

    # initial version
    #def _wf(itimes):
    #    c, _ = np.histogram(itimes, bins = wftimes)
    #    return c

    wfs =  np.array([_wf(_times(ts, ipes)) for ipes in pes.T])

    return wftimes, wfs

create_wfs_pmts= create_wfs

def create_wfs_sipms(ts, pes,
                         wf_buffer_time : float = detsimparams.wf_buffer_time,
                         wf_bin_time    : float = detsimparams.wf_sipm_bin_time,
                         el_time_sample : float = detsimparams.EL_sipm_time_sample,
                         el_time        : float = detsimparams.EL_dtime):
    return create_wfs(ts, pes, wf_buffer_time, wf_bin_time, el_time_sample, el_time)


#
#  Plotting
#-----------------

def histo(var, name = '', nbins = 100):
    ht.hist(var, nbins)
    plt.xlabel(name)

#
# Drivers
#-----------------

def generate_wfs(xs     : np.array,
                 ys     : np.array,
                 zs     : np.array,
                 enes   : np.array,
                 histos : bool = False) -> Tuple:

    t0 = time.time()
    nes = generate_electrons(enes)
    if (histos):
        print('number of secondary elctrons ', np.sum(nes))
        histo(nes, 'number of electrons ')

    # drift electrons
    nes = drift_electrons(zs, nes)
    if (histos):
        print('number of drifted electrons ', np.sum(nes));
        histo(nes, 'number of driffer electrons ')

    # diffuse electrons
    dxs, dys, dzs, dnes = diffuse_electrons(xs, ys, zs, nes)
    dts                 = dzs / detsimparams.drift_velocity
    if (histos):
        print('number of diffused electrons ', np.sum(nes));
        print('longitudinal diffusion ', detsimparams.longitudinal_diffusion * np.sqrt(np.mean(zs)))
        print('transverse diffusion'   , detsimparams.transverse_diffusion * np.sqrt(np.mean(zs)))
        histo(dxs            , 'diffused electrons x (mm)')
        histo(dys            , 'diffused electrons y (mm)')
        histo(dzs            , 'diffused electrons z (mm)')
        histo(dts / units.mus, 'diffused electrons time (ms)')

    t1 = time.time()
    # generate EL photons
    photons = generate_EL_photons(dnes)
    if (histos):
        print('number of EL photons', np.sum(photons))
        histo(photons, 'photons per electron')

    # estimate pes at PMTs
    pes_pmts = estimate_pes_at_pmts(dxs, dys, photons)
    if (histos):
        print('pes ', np.sum(pes_pmts, axis = 0))
        histo(pes_pmts, 'pes at pmts')

    # estimate pes at SiPMs
    pes_sipms = estimate_pes_at_sipms(dxs, dys, photons)
    if (histos):
        print('pes ', np.sum(pes_sipms, axis = 0))
        histo(pes_sipms, 'pes at sipms')

        plt.figure();
        int_pes_sipms  = np.sum(pes_sipms, axis = 0)
        plt.hist2d(detsimparams.x_sipms, detsimparams.y_sipms, bins = detsimparams.xybins, weights = int_pes_sipms);

    t2 = time.time()
    # WFs PMTs
    times_pmts, wfs_pmts = create_wfs_pmts(dts, pes_pmts)
    if (histos):
        xcenters = times_pmts
        plt.figure()
        plt.plot(xcenters / units.mus, wfs_pmts.T); plt.xlabel('wfs pmts')
        plt.xlim((np.min(dts) /units.mus - 5, np.max(dts) / units.mus + 10));

    # WFS SiPMs
    times_sipms, wfs_sipms = create_wfs_sipms(dts, pes_sipms)
    if (histos):
        xcenters = times_sipms
        plt.figure()
        plt.plot(xcenters / units.mus, wfs_sipms.T); plt.xlabel('wfs sipms')
        plt.xlim((np.min(dts) /units.mus - 5, np.max(dts) / units.mus + 10));

    t3 = time.time()
    if (histos):
        print('timing generate, pes, wfs', t1 - t0, t2 - t1, t3 - t2, ' s')
    return (times_pmts, wfs_pmts), (times_sipms, wfs_sipms)
