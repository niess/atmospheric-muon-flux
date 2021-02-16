#! /usr/bin/env python3
import argparse
import numpy
import os

from MCEq.core import config, MCEqRun
import crflux.models as crf


def generate_table(interaction_model=None, primary_model=None,
    density_model=None):

    interaction_model = interaction_model or 'SIBYLL23C'
    primary_model = primary_model or 'H3a'
    density_model = density_model or 'USStd'
    tag = '-'.join((interaction_model.lower(), primary_model.lower(),
        density_model.lower()))

    weights = None
    if interaction_model == 'YFM':
        # Use weights from Yanez et al., 2019 (https://arxiv.org/abs/1909.08365)
        interaction_model = 'SIBYLL23C'
        weights = {211: 0.141, -211: 0.116, 321: 0.402, -321: 0.583}

    if primary_model == 'GSF':
        primary_model = (crf.GlobalSplineFitBeta, None)
    elif primary_model == 'H3a':
        primary_model = (crf.HillasGaisser2012, 'H3a')
    elif primary_model == 'PolyGonato':
        primary_model = (crf.PolyGonato, None)
    else:
        raise ValueError(f'Invalid primary model: {primary_model}')

    if density_model == 'USStd':
        density_model = ('CORSIKA', ('USStd', None))
    elif density_model.startswith('MSIS00'):
        density_model = ('MSIS00', density_model.split('-')[1:])
    else:
        raise ValueError(f'Invalid density model: {density_model}')

    config.e_min = 1E-01
    config.enable_default_tracking = False
    config.enable_muon_energy_loss = True

    mceq = MCEqRun(
        interaction_model = interaction_model,
        primary_model = primary_model,
        density_model = density_model,
        theta_deg = 0
    )

    if weights:
        def weight(xmat, egrid, name, c):
            return (1 + c) * numpy.ones_like(xmat)

        for pid, w in weights.items():
            mceq.set_mod_pprod(2212, pid, weight, ('a', w))
        mceq.regenerate_matrices(skip_decay_matrix=True)

    energy = mceq.e_grid
    cos_theta = numpy.linspace(0, 1, 51)
    altitude = numpy.linspace(0, 9E+03, 10)

    data = numpy.zeros((altitude.size, cos_theta.size, energy.size, 2))
    for ic, ci in enumerate(cos_theta):
        print(f'processing {ci:.2f}')

        theta = numpy.arccos(ci) * 180 / numpy.pi
        mceq.set_theta_deg(theta)
        X_grid = mceq.density_model.h2X(altitude[::-1] * 1E+02)
        mceq.solve(int_grid=X_grid)

        for index, _ in enumerate(altitude):
            mu_m = mceq.get_solution('mu-', grid_idx=index) * 1E+04
            mu_p = mceq.get_solution('mu+', grid_idx=index) * 1E+04
            K = (mu_m > 0) & (mu_p > 0)
            data[altitude.size - 1 - index,ic,K,0] = mu_m[K]
            data[altitude.size - 1 - index,ic,K,1] = mu_p[K]

    # Dump the data grid to a litle endian binary file
    data = data.astype('f4').flatten()
    with open(f'data/simulated/flux-mceq-{tag}.table', 'wb') as f:
        numpy.array((energy.size, cos_theta.size, altitude.size),
            dtype='i8').astype('<i8').tofile(f)
        numpy.array((energy[0], energy[-1], cos_theta[0], cos_theta[-1],
            altitude[0], altitude[-1]), dtype='f8').astype('<f8').tofile(f)
        data.astype('<f4').tofile(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='generate a tabulation of the atmospheric muon flux using MCEq')
    parser.add_argument('-d', '--density-model', type=str,
        help='atmosphere density model')
    parser.add_argument('-i', '--interaction-model', type=str,
        help='hadronic interaction model')
    parser.add_argument('-p', '--primary-model', type=str,
        help='primaries composition model')

    args = vars(parser.parse_args())
    generate_table(**args)
