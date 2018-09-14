#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A class to compute surface wave velocity maps from crustal thickness.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2015
    Simon Stähler, 2018
:license:
    None
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import re
from scipy import interpolate
import shtns
import specnm
import tempfile
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from h5py import File

def _find_cmb_layer(vs):
    for i in range(0, len(vs)):
        if vs[i] < 1 and vs[i + 1] > 1:
            break
    return i


class CrustalThicknessToVelocity(object):
    """
    A class to compute surface wave velocity maps from crustal thickness.
    """

    def __init__(self, master_model_file, path_out='.',
                 nthick=10, mtype='rayleigh', overtone_number=0, fmin=0.005,
                 fmax=0.1, nfreq=100, intk=3, with_topo=False):

        with File(master_model_file, 'r') as h5file:
            self.min_thickness = max(np.asarray(h5file['status']['thickness_minimum']), 10100.)
            # print('Minimum thickness: %5.1f m' % self.min_thickness)
            self.max_thickness = np.asarray(h5file['status']['thickness_maximum'])
            # print('Maximum thickness: %5.1f m' % self.max_thickness)
            self.mtype = mtype
            self.overtone_number = overtone_number
            self.fmin = fmin * 0.99
            self.fmax = fmax * 1.01

            ts = np.linspace(self.min_thickness,
                             self.max_thickness,
                             nthick, endpoint=True)
            fs = np.linspace(fmin, fmax, nfreq, endpoint=True)

            vp = np.zeros((nfreq, nthick))
            vg = np.zeros((nfreq, nthick))

            model = CrustalThicknessToVelocity.read_hdf5_model(h5file=h5file,
                                                               thicknesses=ts,
                                                               path_out=path_out)

        for i, t in enumerate(tqdm(ts)):
            sw = specnm.SurfaceWaveDispersion(
                model[i], mtype=mtype, min_overtone_number=overtone_number, l0=25,
                shortest_period=1. / fmax)

            vp[:, i] = sw.phase_velocity(
                1. / fs, overtone_number=overtone_number)
            vg[:, i] = sw.group_velocity(
                1. / fs, overtone_number=overtone_number)

            # os.remove(model)

        self.__phase_velocity = interpolate.RectBivariateSpline(
            fs, ts, vp, kx=intk, ky=intk)
        self.__group_velocity = interpolate.RectBivariateSpline(
            fs, ts, vg, kx=intk, ky=intk)

    def __check_frequencies(self, frequencies):
        if (np.min(frequencies) < self.fmin or
                np.max(frequencies) > self.fmax):
            raise ValueError(
                'period outside the range covered by this instance')

    def phase_velocity(self, period, thickness):
        self.__check_frequencies(1. / period)
        velo = self.__phase_velocity(1. / period, thickness, grid=False)
        return velo

    def group_velocity(self, period, thickness):
        self.__check_frequencies(1. / period)
        velo = self.__group_velocity(1. / period, thickness, grid=False)
        return velo

    @staticmethod
    def read_hdf5_model(h5file, thicknesses, path_out):
        mantle = h5file['mantle']
        fnam_model = h5file.filename
        nlayer = len(mantle['vp'])
        vp = np.asarray(mantle['vp'])
        vs = np.asarray(mantle['vs'])
        rho = np.asarray(mantle['rho'])
        qmu = np.asarray(mantle['qmu'])
        qka = np.asarray(mantle['qka'])
        radius = np.asarray(mantle['radius'])

        lower = {'bedrock': -2,
                 'uc': -4,
                 'lc': -6}
        upper = {'bedrock': -1,
                 'uc': -3,
                 'lc': -5,
                 'mantle': -7}
        crust_vp = dict()
        crust_vs = dict()
        crust_rho = dict()
        crust_qka = dict()
        crust_qmu = dict()
        for lay in ['bedrock', 'uc', 'lc']:
            crust_vp[lay] = mantle['vp'][lower[lay]]
            crust_vs[lay] = mantle['vs'][lower[lay]]
            crust_rho[lay] = mantle['rho'][lower[lay]]
            crust_qka[lay] = mantle['qka'][lower[lay]]
            crust_qmu[lay] = mantle['qmu'][lower[lay]]

        radius_planet = radius[-1]
        radius_bedrock = radius[-3]
        radius_conrad = radius[-5]

        # Layers below Moho are at 80 km, 100km, 110km, 120km...
        # At 80 km there is also a discontinuity. Because YOLO!
        istart = upper['mantle'] - 6  # lower['lc'] - 3
        iend = upper['mantle'] + 1 # lower['lc'] - 1

        ipl_mantle = dict()

        for var in ['vp', 'vs', 'rho', 'qka', 'qmu']:
            ipl_mantle[var] = interpolate.interp1d(
                x=radius[istart:iend],
                y=mantle[var][istart:iend],
                kind='linear',
                fill_value='extrapolate', assume_sorted=True)

        line_fmt = '%8.0f %8.2f %8.2f %8.2f %9.1f %9.1f %8.2f %8.2f 1.0\n'
        fnams = []
        os.makedirs(os.path.join(path_out, 'modified_models'), exist_ok=True)

        for crustal_thickness in thicknesses:
            radius_moho = radius_planet - crustal_thickness
            # Need to find layer, where new Moho is and omit everything
            # above that
            upper['mantle_new'] = min(upper['mantle'], - (nlayer - np.argmax(radius > radius_moho)))
            fnam = os.path.join(path_out,
                                'modified_models',
                                os.path.splitext(fnam_model)[0] +
                                '_%05.1f' % crustal_thickness + '.deck')
            fnams.append(fnam)

            with open(fnam, 'w') as f:
                f.write('Original model: %s, new crustal thickness: %4.1fkm\n' %
                        (fnam_model, crustal_thickness))
                f.write('0 1.0 1\n')
                f.write('%d %d %d %d\n' % (nlayer, 0,
                                           _find_cmb_layer(mantle['vs']) + 1, nlayer - 7))
                for i in range(0, nlayer + upper['mantle_new']):
                    f.write(line_fmt %
                            (radius[i],
                             rho[i],
                             vp[i],
                             vs[i],
                             qka[i],
                             qmu[i],
                             vp[i],
                             vs[i]))

                # Write line below Moho
                f.write(line_fmt %
                        (radius_moho,
                         ipl_mantle['rho'](radius_moho),
                         ipl_mantle['vp'](radius_moho),
                         ipl_mantle['vs'](radius_moho),
                         ipl_mantle['qmu'](radius_moho),
                         ipl_mantle['qka'](radius_moho),
                         ipl_mantle['vp'](radius_moho),
                         ipl_mantle['vs'](radius_moho))
                        )

                # Write line above Moho
                f.write(line_fmt %
                        (radius_moho,
                         crust_rho['lc'],
                         crust_vp['lc'],
                         crust_vs['lc'],
                         crust_qmu['lc'],
                         crust_qka['lc'],
                         crust_vp['lc'],
                         crust_vs['lc']))

                # Write line below Conrad
                f.write(line_fmt %
                        (radius_conrad,
                         crust_rho['lc'],
                         crust_vp['lc'],
                         crust_vs['lc'],
                         crust_qmu['lc'],
                         crust_qka['lc'],
                         crust_vp['lc'],
                         crust_vs['lc']))
                # Write line above Conrad
                f.write(line_fmt %
                        (radius_conrad,
                         crust_rho['uc'],
                         crust_vp['uc'],
                         crust_vs['uc'],
                         crust_qmu['uc'],
                         crust_qka['uc'],
                         crust_vp['uc'],
                         crust_vs['uc']))

                # Write line below bedrock
                f.write(line_fmt %
                        (radius_bedrock,
                         crust_rho['uc'],
                         crust_vp['uc'],
                         crust_vs['uc'],
                         crust_qmu['uc'],
                         crust_qka['uc'],
                         crust_vp['uc'],
                         crust_vs['uc']))
                # Write line above bedrock
                f.write(line_fmt %
                        (radius_bedrock,
                         crust_rho['bedrock'],
                         crust_vp['bedrock'],
                         crust_vs['bedrock'],
                         crust_qmu['bedrock'],
                         crust_qka['bedrock'],
                         crust_vp['bedrock'],
                         crust_vs['bedrock']))

                # Write line at surface
                f.write(line_fmt %
                        (radius_planet,
                         crust_rho['bedrock'],
                         crust_vp['bedrock'],
                         crust_vs['bedrock'],
                         crust_qmu['bedrock'],
                         crust_qka['bedrock'],
                         crust_vp['bedrock'],
                         crust_vs['bedrock']))

        return fnams

    @staticmethod
    def adjust_crustal_thickness_deck(model_file, crustal_thickness,
                                      new_model_file=None, with_topo=False):

        with open(model_file, 'r') as f:
            lines = f.readlines()

        radius = float(lines[-1].split()[0])
        nextlayer = float(lines[-4].split()[0])
        moho_bg = float(lines[-2].split()[0])
        moho = radius - crustal_thickness
        crustal_thickness_bg = radius - moho_bg

        if radius - nextlayer <= crustal_thickness:
            raise ValueError('new crust too thick')

        for ll in [-2, -3]:
            lsplit = re.split(r'(\s+)', lines[ll])
            lsplit[0] = '%7.0f.' % (moho,)

            buf = ''.join(lsplit)
            lines[ll] = buf

        # move surface along with moho
        if with_topo:
            ll = -1
            lsplit = re.split(r'(\s+)', lines[ll])
            lsplit[0] = '%7.0f.' % (moho + crustal_thickness_bg,)
            buf = ''.join(lsplit)
            lines[ll] = buf

        if new_model_file:
            f = open(new_model_file, 'w')
        else:
            f = tempfile.NamedTemporaryFile(delete=False, suffix='_sw')

        f.writelines(lines)
        return f.name

    def logspaced_periods(self, nsamp=50):
        return np.logspace(np.log10(1. / (self.fmin * 1.01)),
                           np.log10(1. / (self.fmax * 0.99)),
                           nsamp)

    def plot_phase_velocity_map(self, period, thickness, mode='both',
                                show=True, cmap="RdBu", relative=False):

        if mode == 'both':
            velo = ['group', 'phase']
        else:
            velo = [mode]
        velo_map = {'group': self.group_velocity,
                    'phase': self.phase_velocity}

        figs = []
        for v in velo:
            fig = plt.figure()
            ax = plt.gca()
            vmap = velo_map[v](period, thickness)

            if relative:
                ntheta = thickness.shape[1]
                nphi = thickness.shape[0]
                theta, dtheta = np.linspace(0., 180., ntheta, False, True)
                theta += dtheta / 2.
                theta = theta.repeat(nphi).reshape(ntheta, nphi).T
                theta = np.deg2rad(theta)

                # mean of the velocity
                # vmean = (vmap * np.sin(theta)).sum() / (np.sin(theta)).sum()

                # velocity of the mean crustal thickness
                tmean = (thickness * np.sin(theta)).sum() / \
                        (np.sin(theta)).sum()
                vmean = velo_map[v](period, tmean)

                vmean2, std = weighted_avg_and_std(vmap, np.sin(theta))

                vmean_reldiff = (vmean - vmean2) / (vmean + vmean2) * 2 * 100

                vmap = (vmap - vmean) / vmean * 100
                vmax = np.max(np.abs(vmap))
                vmin = -vmax
                # vmin, vmax = None, None
                cblabel = '%s velocity perturbation / %%' % (v,)
                plt.title('%s velocity at %ds, velocity at mean crustal '
                          'thickness = %3.1f km / s' %
                          (v, period, vmean))
            else:
                vmin, vmax = None, None
                cblabel = '%s velocity / (km / s)' % (v,)
                plt.title('%s velocity at %ds' % (v, period))

            im = ax.imshow(vmap.T, extent=[0., 360., -90., 90.], cmap=cmap,
                           vmin=vmin, vmax=vmax)

            plt.xlim(0., 360.)
            plt.ylim(-90., 90.)
            ax.set_xticks([0., 90., 180., 270., 360.])
            ax.set_yticks([-90., -45., 0., 45, 90.])
            plt.xlabel('longitude')
            plt.ylabel('latitude')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.2)
            plt.colorbar(im, cax=cax, label=cblabel)
            figs.append(fig)

        if show:
            plt.show()
        else:
            return figs

    def plot(self, nperiods=10, mode='both', ntsamp=100, show=True):
        t = np.linspace(self.min_thickness, self.max_thickness, ntsamp)

        if mode == 'both':
            velo = ['group', 'phase']
        else:
            velo = [mode]
        velo_map = {'group': self.group_velocity,
                    'phase': self.phase_velocity}

        figs = []
        for v in velo:
            fig = plt.figure()
            for p in self.logspaced_periods(nperiods):
                plt.plot(t / 1e3, velo_map[v](p, t), label='%d s' % (p,))
            plt.xlabel('crustal thickness / km')
            plt.ylabel('%s velocity / (km / s)' % (v,))
            plt.title('%s velocity' % (v,))
            plt.legend(fancybox=True, framealpha=0.7)
            figs.append(fig)

        if show:
            plt.show()
        else:
            return figs

    def plot_dispersion_curves(self, nperiods=100, mode='both', ntsamp=10,
                               show=True):

        t = np.linspace(self.min_thickness, self.max_thickness, ntsamp)
        p = self.logspaced_periods(nperiods)

        if mode == 'both':
            velo = ['group', 'phase']
        else:
            velo = [mode]
        velo_map = {'group': self.group_velocity,
                    'phase': self.phase_velocity}

        figs = []
        for v in velo:
            fig = plt.figure()
            for i, _t in enumerate(t):
                plt.plot(p, velo_map[v](p, _t), label='%d km' % (_t / 1e3,))
            plt.xlabel('period / s')
            plt.ylabel('%s velocity / (km / s)' % (v,))
            plt.title('%s velocity' % (v,))
            plt.legend(fancybox=True, framealpha=0.7, loc='best')
            figs.append(fig)

        if show:
            plt.show()
        else:
            return figs

    def plot_velocity_bias(self, thickness, nperiods=100, mode='both',
                           show=True):

        periods = self.logspaced_periods(nperiods)

        if mode == 'both':
            velo = ['group', 'phase']
        else:
            velo = [mode]
        velo_map = {'group': self.group_velocity,
                    'phase': self.phase_velocity}

        figs = []
        for v in velo:
            vmean = np.zeros(nperiods)
            vhmean = np.zeros(nperiods)
            vstd = np.zeros(nperiods)

            ntheta = thickness.shape[1]
            nphi = thickness.shape[0]
            theta, dtheta = np.linspace(0., 180., ntheta, False, True)
            theta += dtheta / 2.
            theta = theta.repeat(nphi).reshape(ntheta, nphi).T
            theta = np.deg2rad(theta)

            # velocity of the mean crustal thickness
            tmean, tstd = weighted_avg_and_std(thickness, np.sin(theta))
            vctmean = velo_map[v](periods, tmean)

            for i, p in enumerate(periods):
                vmap = velo_map[v](p, thickness)

                # mean and std of the velocity
                vmean[i], vstd[i] = weighted_avg_and_std(vmap, np.sin(theta))
                vhmean[i] = 1. / weighted_avg_and_std(1. / vmap, np.sin(theta))[0]

            '''
            from scipy import interpolate
            p = self.logspaced_periods(5)

            f = interpolate.interp1d(periods, vctmean)
            print(v, 'ctmean')
            for _p in p:
                print(_p, f(_p))
            f = interpolate.interp1d(periods, vmean)
            print(v, 'mean')
            for _p in p:
                print(_p, f(_p))
            print(v, 'hmean')
            f = interpolate.interp1d(periods, vhmean)
            for _p in p:
                print(_p, f(_p))
            '''
            figs.append(plt.figure())
            plt.plot(periods, vhmean, 'k', label='1/<1/v(ct)>')
            plt.plot(periods, vmean, 'b', label='<v(ct)>')
            plt.plot(periods, vctmean, 'r', label='v(<ct>)')

            plt.title('%s velocity' % (v,))
            plt.legend(loc='best')
            plt.xlabel('period / s')
            plt.ylabel('velocity / (km / s)')

            figs.append(plt.figure())
            plt.plot(periods,
                     (vctmean - vhmean) / (vhmean + vctmean) * 2 * 100, 'k',
                     label='relative averaging bias, hmean')
            plt.plot(periods,
                     (vctmean - vmean) / (vmean + vctmean) * 2 * 100, 'b',
                     label='relative averaging bias, mean')
            plt.plot(periods, vstd / vmean * 100, 'r',
                     label='coefficient of variation')

            plt.title('%s velocity' % (v,))
            plt.legend(loc='best')
            plt.xlabel('period / s')
            plt.ylabel('%')

        if show:
            plt.show()
        else:
            return figs


def read_crustal_thickness_h5(model_file):
    with File(model_file, 'r') as d:
        crust = d['crust']
        moho = np.asarray(crust['moho'])
        topo = np.asarray(crust['topo'])
        lats = np.asarray(crust['latitudes'])
        lons = np.asarray(crust['longitudes'])

    return topo, moho, lats, lons


def read_crustal_thickness_wieczorek(fname, shape=(181, 360)):
    # read crustal thickness models in the xyz format by Marc Wieczorek
    data = np.loadtxt(fname)
    thickness = data[:, 4]
    thickness = thickness.reshape(shape).T

    # interpolate to latitude grid that does not include the poles (as
    # needed by shtns)
    thickness = (thickness[:, :-1] + thickness[:, 1:]) / 2

    return thickness


def filter_model_shtns(model, lmax, nphi_out=None, ntheta_out=None, order=2,
                       lmax_transform=None):
    # assumes the model is sampled on a regular grid where the poles are not
    # included

    if lmax_transform is None:
        lmax_transform = lmax * 2

    nphi, ntheta = model.shape
    lmax_transform = min(min(ntheta - 2, nphi / 2 - 1), lmax_transform)

    # prepare the transform
    mmax = lmax_transform
    sh = shtns.sht(lmax_transform, mmax)

    grid_typ = shtns.sht_reg_fast | shtns.SHT_THETA_CONTIGUOUS
    grid_typ = grid_typ | shtns.SHT_LOAD_SAVE_CFG

    polar_opt_threshold = 1.0e-10
    sh.set_grid(ntheta, nphi, flags=grid_typ,
                polar_opt=polar_opt_threshold)

    # apply transform
    ylm = sh.analys(model)

    ylm *= 1. / (1. + (sh.l * 1. / lmax) ** (2 * order))

    if nphi_out and ntheta_out:
        sh.set_grid(ntheta_out, nphi_out, flags=grid_typ,
                    polar_opt=polar_opt_threshold)

    return sh.synth(ylm)


def write_model_h5(model, lmax, fname, varpath):
    # assumes the model is sampled on a regular grid where the poles are not
    # included

    # prepare the transform
    mmax = lmax
    sh = shtns.sht(lmax, mmax)

    npts_ph, npts_th = model.shape

    grid_typ = shtns.sht_reg_fast | shtns.SHT_THETA_CONTIGUOUS
    grid_typ = grid_typ | shtns.SHT_LOAD_SAVE_CFG

    polar_opt_threshold = 1.0e-10
    sh.set_grid(npts_th, npts_ph, flags=grid_typ,
                polar_opt=polar_opt_threshold)

    # apply transform
    ylm = sh.analys(model)

    with File(fname, 'r+') as f:
        path_l = os.path.split(varpath)[0] + '/l'
        if not path_l in f:
            f.create_dataset(path_l, data=sh.l)
        path_m = os.path.split(varpath)[0] + '/m'
        if not path_m in f:
            f.create_dataset(path_m, data=sh.m)
        f.create_dataset(varpath, data=ylm)

    return


def write_model_ylm(model, lmax, fname):
    # assumes the model is sampled on a regular grid where the poles are not
    # included

    # prepare the transform
    mmax = lmax
    sh = shtns.sht(lmax, mmax)

    npts_ph, npts_th = model.shape

    grid_typ = shtns.sht_reg_fast | shtns.SHT_THETA_CONTIGUOUS
    grid_typ = grid_typ | shtns.SHT_LOAD_SAVE_CFG

    polar_opt_threshold = 1.0e-10
    sh.set_grid(npts_th, npts_ph, flags=grid_typ,
                polar_opt=polar_opt_threshold)

    # apply transform
    ylm = sh.analys(model)

    with open(fname, 'w') as f:
        for l in zip(sh.l, sh.m, np.real(ylm), np.imag(ylm)):
            f.write('%5d %5d %14e %14e\n' % l)
    return


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, math.sqrt(variance))

