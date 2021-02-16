#! /usr/bin/env python3
import argparse
import collections
import matplotlib.pyplot as plot
import numpy
import sys


'''Container for experimental flux data
'''
ExperimentalData = collections.namedtuple(
    'experimentaldata', ('energy', 'flux', 'dflux', 'ratio', 'dratio'))


def load_table(path):
    '''Load simulated flux data from a binary table
    '''
    with open(path, 'rb') as f:
        shape = numpy.fromfile(f, '<i8', 3)
        ranges = numpy.fromfile(f, '<f8', 6)
        data = numpy.fromfile(f, '<f4', 2 * shape[0] * shape[1] * shape[2])
        data = data.reshape((shape[2], shape[1], shape[0], 2))

        energy = numpy.logspace(
            numpy.log10(ranges[0]), numpy.log10(ranges[1]), shape[0])
        cos_theta = numpy.linspace(ranges[2], ranges[3], shape[1])
        altitude = numpy.linspace(ranges[4], ranges[5], shape[2])

    return energy, cos_theta, altitude, data


def load_bess():
    '''Load and format BESS-TeV data
    '''
    data = numpy.loadtxt('data/measured/BESS_TEV.txt', comments='#')

    m = 0.10566                     # Muon mass (GeV/c^2)
    p = data[:,2]                   # Momentum (GeV)
    e = numpy.sqrt(p**2 + m**2) - m # Kinetic energy (GeV)
    j = (e + m) / p                 # Jacobian factor for going from dphi / dp
                                    #   to dphi / dE
    f = (data[:,3] + data[:,7]) * j # Flux (1/(GeV m^2 s sr))

    df = numpy.sqrt(data[:,4]**2 + data[:,5]**2 + # Flux uncertainty
                    data[:,8]**2 + data[:,9]**2) * j

    r = data[:,3] / data[:,7] # Charge ratio and uncertainty
    dr = r * numpy.sqrt(
        (data[:,4]**2 + data[:,5]**2) / data[:,3]**2 +
        (data[:,8]**2 + data[:,9]**2) / data[:,7]**2)

    return ExperimentalData(e, f, df, r, dr)


def load_L3C():
    '''Load and format L3-Cosmic data
    '''
    data = numpy.loadtxt('data/measured/L3+C.txt', comments = '#')

    m = 0.10566   # Muon mass (GeV/c^2)
    p = data[:,2] # Momentum (GeV)
    e = numpy.sqrt(p**2 + m**2) - m   # Kinetic energy (GeV)
    j = (e + m) / p                   # Jacobian factor for going from dphi / dp
                                      #   to dphi / dE
    f = 1E+04 * data[:,3] / p**3 * j  # Flux (1 /(GeV m^2 s sr))
    df = f * numpy.sqrt(data[:,4]**2 + data[:,6]**2) * 1E-02 * j # Flux uncert.
    r = data[:,7] # Charge ratio and uncertainty
    dr = r * numpy.sqrt(data[:,8]**2 + data[:,10]**2) * 1E-02

    return ExperimentalData(e, f, df, r, dr)


def plot_table(path):
    '''Plot simulated flux data from a binary table
    '''

    energy, cos_theta, altitude, data = load_table(path)

    bess = load_bess()
    f95 = numpy.mean(data[0,-4:-2,:, :], axis=0)  # Total flux at
                                                  # cos(theta) = 0.95 and 0m

    l3c = load_L3C()
    f10 = numpy.mean(data[:2,-2:,:, :], axis=(0, 1)) # Total flux at
                                                     # cos(theta) = 0.99
                                                     #and 500m

    plot.figure()
    plot.loglog(energy, energy**3 * numpy.sum(f95, axis=1), 'k--')
    plot.errorbar(bess.energy, bess.energy**3 * bess.flux,
        yerr=bess.energy**3 * bess.dflux, fmt='bo', label='BESS-TeV')

    plot.loglog(energy, energy**3 * numpy.sum(f10, axis=1), 'k--')
    plot.errorbar(l3c.energy, l3c.energy**3 * l3c.flux,
        yerr=l3c.energy**3 * l3c.dflux, fmt='ro', label='L3+C')
    plot.xlabel('energy [GeV]')
    plot.ylabel('flux (E / GeV)$^3$ [GeV$^{-1}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]')
    plot.axis((1E-01, 1E+04, 1E+00, 1E+04))
    plot.legend(loc=4)

    plot.figure()
    plot.semilogx(energy, f95[:,1] / f95[:,0], 'k-')
    plot.errorbar(bess.energy, bess.ratio, yerr=bess.dratio, fmt='bo',
        label='BESS-TeV')
    plot.errorbar(l3c.energy, l3c.ratio, yerr=l3c.dratio, fmt='ro',
        label='L3+C')
    plot.xlabel('energy [GeV]')
    plot.ylabel('charge ratio ($\\mu^+$ / $\\mu^-$)')
    plot.axis((1E-01, 1E+04, 0.5, 2.0))
    plot.legend(loc=4)

    plot.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='plot a tabulation of the atmospheric muon flux')
    parser.add_argument('path', type=str, help='path to the tabulation')

    args = parser.parse_args()
    plot_table(args.path)
