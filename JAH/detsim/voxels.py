import numpy             as np
import scipy             as sc
import scipy.stats       as st
import tables            as tb

def equal_size_bins(xmin, xmax, dx):
    nbins   = int((xmax - xmin)/dx) + 1
    return np.linspace(xmin, xmin + nbins *  dx, nbins + 1)

def bins_centers(edges):
    return 0.5*(edges[1:] + edges[:-1])

def bins_edges(centers, dx):
    bins = centers - 0.5 * dx
    return np.append(bins, bins[-1] + dx)

def masks_in_ranges(values, ranges):
    nbins = len(ranges)
    sels = [(values >= ranges[i]) & (values < ranges[i+1]) for i in range(nbins-1)]
    sels[-1] = (values >= ranges[-2]) & (values <= ranges[-1])
    sels = sels if len(sels) > 1 else sels[0]
    return sels

def voxels1d(xs, xbins):
    contents, xedges = np.histogram(xs, bins = xbins)
    xcenters = bins_centers(xedges)
    return (xcenters, contents)
        
def voxels2d(xs, ys, xbins, ybins):
    contents, xedges, yedges  = np.histogram2d(xs, ys, bins = (xbins, ybins))
    xcenters = bins_centers(xedges)
    ycenters = bins_centers(yedges)
    ymesh, xmesh = np.meshgrid(ycenters, xcenters)
    sel      = contents > 0
    vxs = xmesh[sel]   .flatten()
    vys = ymesh[sel]   .flatten()
    vcs = contents[sel].flatten()
    return (vxs, vys, vcs)

def voxels3d(xs, ys, zs, xbins, ybins, zbins):
    #zbins    = equal_size_bins(np.min(zs), np.max(zs), dz)
    zsels    = masks_in_ranges(zs, zbins)
    zcenters = bins_centers(zbins)
    vxs, vys, vzs, vcs = [], [], [], [] 
    for i, isel in enumerate(zsels):
        if (np.sum(isel) <= 0): continue
        ixs, iys, ics = voxels2d(xs[isel], ys[isel], xbins, ybins)
        vxs.append(ixs)
        vys.append(iys)
        vzs.append(np.repeat(zcenters[i], len(ixs)))
        vcs.append(ics)
    return (np.concatenate(vxs), np.concatenate(vys), np.concatenate(vzs), np.concatenate(vcs))