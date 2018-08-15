#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A class to compute surface wave velocity maps from crustal thickness.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2015
:license:
    None
"""
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import re
from scipy import interpolate
import shtns
# import swdisp
import tempfile
from mpl_toolkits.axes_grid1 import make_axes_locatable


class CrustalThicknessToVelocity(object):
    """
    A class to compute surface wave velocity maps from crustal thickness.
    """
    def __init__(self, master_model_file, min_thickness=0., max_thickness=1e5,
                 nthick=10, mtype='rayleigh', overtone_number=0, fmin=0.005,
                 fmax=0.1, nfreq=100, intk=3, with_topo=False):

        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.mtype = mtype
        self.overtone_number = overtone_number
        self.fmin = fmin
        self.fmax = fmax

        ts = np.linspace(min_thickness, max_thickness, nthick, endpoint=True)
        fs = np.linspace(fmin, fmax, nfreq, endpoint=True)

        vp = np.zeros((nfreq, nthick))
        vg = np.zeros((nfreq, nthick))

        for i, t in enumerate(ts):
            model = CrustalThicknessToVelocity.adjust_crustal_thickness_deck(
                master_model_file, t, with_topo=with_topo, new_model_file=master_model_file + '_radius_%03d' % i)
            sw = swdisp.SurfaceWaveDispersion(
                model, mtype=mtype, min_overtone_number=overtone_number,
                shortest_period=1./fmax)

            vp[:, i] = sw.phase_velocity(
                1. / fs, overtone_number=overtone_number)
            vg[:, i] = sw.group_velocity(
                1. / fs, overtone_number=overtone_number)

            #os.remove(model)

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
            raise ValueError('new crust to thick')

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
        return np.logspace(np.log10(1. / self.fmin), np.log10(1. / self.fmax),
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
            print(v, period)
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
                tmean = (thickness * np.sin(theta)).sum() /\
                    (np.sin(theta)).sum()
                vmean = velo_map[v](period, tmean)

                vmean2, std = weighted_avg_and_std(vmap, np.sin(theta))

                vmean_reldiff = (vmean - vmean2) / (vmean + vmean2) * 2 * 100

                print(vmean, vmean2, std / vmean2 * 100, vmean_reldiff)

                vmap = (vmap - vmean) / vmean * 100
                vmax = np.max(np.abs(vmap))
                vmin = -vmax
                #vmin, vmax = None, None
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

            figs.append(plt.figure())
            plt.plot(periods, vhmean, 'k', label='1/<1/v(ct)>')
            plt.plot(periods, vmean, 'b', label='<v(ct)>')
            plt.plot(periods, vctmean, 'r', label='v(<ct>)')

            plt.title('%s velocity' % (v, ))
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

            plt.title('%s velocity' % (v, ))
            plt.legend(loc='best')
            plt.xlabel('period / s')
            plt.ylabel('%')

        if show:
            plt.show()
        else:
            return figs


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
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))


if __name__ == '__main__':
    ctvelo = CrustalThicknessToVelocity('test_20km', nthick=20, overtone_number=0,
                                        fmin=.003, intk=2, min_thickness=1.,
                                        with_topo=True)
    figs = ctvelo.plot(show=False)
    for fig in figs:
        fig.set_size_inches(6, 6, forward=True)
        fig.tight_layout()
        fig.axes[0].set_xlabel('topo / km')
    figs[0].savefig('group_thick_with_topo.pdf')
    figs[1].savefig('phase_thick_with_topo.pdf')
    plt.show()

    ctvelo = CrustalThicknessToVelocity('test', nthick=20, overtone_number=0,
                                        fmin=.003, intk=2, min_thickness=1.)
    figs = ctvelo.plot(show=False)
    for fig in figs:
        fig.set_size_inches(6, 6, forward=True)
        fig.tight_layout()
    figs[0].savefig('group_thick.pdf')
    figs[1].savefig('phase_thick.pdf')
    plt.show()

    figs = ctvelo.plot_dispersion_curves(show=False)
    for fig in figs:
        fig.set_size_inches(6, 6, forward=True)
        fig.tight_layout()
    figs[0].savefig('group_disp.pdf')
    figs[1].savefig('phase_disp.pdf')
    plt.show()

    thmod = '2900_10_DWTh2Ref1'
    thickness = read_crustal_thickness_wieczorek('%s.xyz' % thmod)

    thickness = filter_model_shtns(thickness, lmax=16, order=6)
    figs = ctvelo.plot_velocity_bias(thickness, show=False)

    for fig in figs:
        fig.set_size_inches(6, 4, forward=True)
        fig.tight_layout()
    figs[0].savefig('group_avg.pdf')
    figs[1].savefig('group_bias.pdf')
    figs[2].savefig('phase_avg.pdf')
    figs[3].savefig('phase_bias.pdf')
    plt.show()

    for p in ctvelo.logspaced_periods(5):
        figs = ctvelo.plot_phase_velocity_map(p, thickness, mode='both',
                                              relative=True, show=False)
        for fig in figs:
            fig.set_size_inches(10, 5, forward=True)
            fig.tight_layout()
        figs[0].savefig('group_map_%03ds.pdf' % p)
        figs[1].savefig('phase_map_%03ds.pdf' % p)
        plt.show()

        v = ctvelo.group_velocity(p, thickness)
        v = filter_model_shtns(v, lmax=16)
        v *= 1e3
        write_model_ylm(v, lmax=32,
                        fname='%s_%03ds_group.ylm' % (thmod, p))

        v = ctvelo.phase_velocity(p, thickness)
        v = filter_model_shtns(v, lmax=16)
        v *= 1e3
        write_model_ylm(v, lmax=32,
                        fname='%s_%03ds_phase.ylm' % (thmod, p))
