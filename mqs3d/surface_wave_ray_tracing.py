#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
A surface wave ray tracing class.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
    Simon Stähler (simonsta@ethz.ch), 2018
:license:
    None
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import shtns
import math
import warnings
from h5py import File


def shoot(latitude_1_degree, longitude_1_degree, bearing_degree, distance_km, radius_km):
    """
    Shoot a ray from point in direction for certain length and return where you land
    (Direct geodetic problem). Works on sphere
    :param latitude_1_degree: latitude of starting point
    :param longitude_1_degree: longitude of starting point
    :param bearing_degree: bearing from north, CW
    :param distance_km: distance in kilometer
    :param radius_km: radius of planet
    :return: latitude, longitude of target
    """
    lat1 = np.deg2rad(latitude_1_degree)
    lon1 = np.deg2rad(longitude_1_degree)
    bearing = np.deg2rad(bearing_degree)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(distance_km / radius_km) +
                     np.cos(lat1) * np.sin(distance_km / radius_km) * np.cos(bearing))
    lon2 = lon1 + np.arctan2(np.sin(bearing) * np.sin(distance_km / radius_km) * np.cos(lat1),
                             np.cos(distance_km / radius_km) - np.sin(lat1) * np.sin(lat2))
    return np.rad2deg(lat2), np.mod(np.rad2deg(lon2) + 540., 360.) - 180.


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))


class SurfaceWaveRayTracer(object):
    def __init__(self, R, lmax, nphi=None, ntheta=None, delta_phi=1., verbose=False):
        self.R = R
        self.lmax = lmax

        # prepare the spherical harmonic transform
        self.nphi = nphi or self.lmax * 4
        self.ntheta = ntheta or self.lmax * 2
        self.mmax = self.lmax
        self.sh = shtns.sht(self.lmax, self.mmax)

        grid_typ = shtns.sht_gauss | shtns.SHT_THETA_CONTIGUOUS | \
            shtns.SHT_LOAD_SAVE_CFG

        polar_opt_threshold = 1.0e-10
        self.sh.set_grid(self.ntheta, self.nphi, flags=grid_typ,
                         polar_opt=polar_opt_threshold)

        self.theta = np.arccos(self.sh.cos_theta)
        self.phi = np.linspace(0, 2 * np.pi, self.nphi)

        self.workspace = self.sh.spec_array()

        self.delta_phi = np.deg2rad(delta_phi)

        self.group_velocity = None
        self.phase_velocity = None
        self.topo = None
        self.group_velocity_ylm = None
        self.phase_velocity_ylm = None
        self.topo_ylm = None
        self.group_velocity_notrot = None
        self.phase_velocity_notrot = None
        self.topo_notrot = None
        self.group_velocity_ylm_notrot = None
        self.phase_velocity_ylm_notrot = None
        self.topo_ylm_notrot = None

    def set_constant_velocity(self, c, which='phase'):
        shape = (self.nphi, self.ntheta)

        if which == 'phase':
            self.phase_velocity = np.ones(shape) * c
            ylm = self.sh.analys(self.phase_velocity.copy())
            self.phase_velocity_ylm_notrot = ylm
            self.phase_velocity_notrot = self.sh.synth(ylm.copy())

        elif which == 'group':
            self.group_velocity = np.ones(shape) * c
            ylm = self.sh.analys(self.group_velocity.copy())
            self.group_velocity_ylm_notrot = ylm
            self.group_velocity_notrot = self.sh.synth(ylm.copy())

        else:
            raise ValueError()

    def load_velocity_model_ylm(self, fname, which='phase'):
        ylm = np.zeros(self.sh.nlm) * 0j

        with open(fname, 'r') as f:
            for line in f:
                ls = line.split()
                l, m = map(int, ls[:2])
                if l > self.lmax:
                    continue
                ylmr, ylmi = map(float, ls[2:])
                # important: map the ylm into the right order of the transform
                #            used here
                ylm[self.sh.idx(l, m)] = ylmr + 1j * ylmi

        if which == 'phase':
            self.phase_velocity_ylm = ylm.copy()
            self.phase_velocity_ylm_notrot = ylm.copy()
            self.phase_velocity = self.sh.synth(ylm).copy()
            self.phase_velocity_notrot = self.phase_velocity.copy()

        elif which == 'group':
            self.group_velocity_ylm = ylm.copy()
            self.group_velocity_ylm_notrot = ylm.copy()
            self.group_velocity = self.sh.synth(ylm).copy()
            self.group_velocity_notrot = self.group_velocity.copy()

        elif which == 'topo':
            self.topo_ylm = ylm.copy()
            self.topo_ylm_notrot = ylm.copy()
            self.topo = self.sh.synth(ylm).copy()
            self.topo_notrot = self.group_velocity.copy()

        else:
            raise ValueError()

    def load_velocity_model_h5(self, fname, type, iperiod):
        with File(fname, 'r') as f:
            ylm = np.zeros(self.sh.nlm) * 0j
            ls = np.asarray(f['ylm_maps/l'], dtype=np.uint)
            ms = np.asarray(f['ylm_maps/m'], dtype=np.uint)

            var_path = 'ylm_maps/tt_%s_phase_%02d' % (type, iperiod)
            ylm_raw = np.asarray(f[var_path])
            for l, m, y in zip(ls, ms, ylm_raw):
                if l > self.lmax:
                    continue
                ylm[self.sh.idx(np.int(l), np.int(m))] = y
            self.phase_velocity_ylm = ylm.copy()
            self.phase_velocity_ylm_notrot = ylm.copy()
            self.phase_velocity = self.sh.synth(ylm).copy()
            self.phase_velocity_notrot = self.phase_velocity.copy()

            var_path = 'ylm_maps/tt_%s_group_%02d' % (type, iperiod)
            ylm_raw = np.asarray(f[var_path])
            for l, m, y in zip(ls, ms, ylm_raw):
                if l > self.lmax:
                    continue
                ylm[self.sh.idx(np.int(l), np.int(m))] = y
            self.group_velocity_ylm = ylm.copy()
            self.group_velocity_ylm_notrot = ylm.copy()
            self.group_velocity = self.sh.synth(ylm).copy()
            self.group_velocity_notrot = self.group_velocity.copy()

            var_path = 'ylm_maps/surface_radius'
            ylm_raw = np.asarray(f[var_path])
            for l, m, y in zip(ls, ms, ylm_raw):
                if l > self.lmax:
                    continue
                ylm[self.sh.idx(np.int(l), np.int(m))] = y
            self.topo_ylm = ylm.copy()
            self.topo_ylm_notrot = ylm.copy()
            self.topo = self.sh.synth(ylm).copy()
            self.topo_notrot = self.group_velocity.copy()

    def compute_mean_velocities(self):

        tm, pm = np.meshgrid(self.theta, self.phi)

        self.mean_phase_velocity = weighted_avg_and_std(
            self.phase_velocity_notrot, weights=np.sin(tm))[0]

        self.hmean_phase_velocity = 1. / weighted_avg_and_std(
            1. / self.phase_velocity_notrot, weights=np.sin(tm))[0]

        if self.group_velocity is not None:
            self.mean_group_velocity = weighted_avg_and_std(
                self.group_velocity_notrot, weights=np.sin(tm))[0]

            self.hmean_group_velocity = 1. / weighted_avg_and_std(
                1. / self.group_velocity_notrot, weights=np.sin(tm))[0]

    def get_grad_phase_velocity(self, theta, phi):
        '''
        compute gradient at a point in the rotated velocity model
        '''
        self.workspace[:] = 0

        gr, gt, gp = shtns.sht.SH_to_grad_point(self.sh, self.workspace,
                                                self.phase_velocity_ylm,
                                                math.cos(theta), phi)

        return gt, gp

    def get_phase_velocity(self, theta, phi):
        '''
        compute phase velocity at a point in the rotated velocity model
        '''
        return shtns.sht.SH_to_point(
            self.sh, self.phase_velocity_ylm, np.cos(theta), phi)

    def get_group_velocity(self, theta, phi):
        '''
        compute group velocity at a point in the rotated velocity model
        '''
        if self.group_velocity is not None:
            return shtns.sht.SH_to_point(
                self.sh, self.group_velocity_ylm, np.cos(theta), phi)
        else:
            return shtns.sht.SH_to_point(
                self.sh, self.phase_velocity_ylm, np.cos(theta), phi)

    def apply_topography_correction(self):
        # waves are faster where the radius is smaller:
        self.phase_velocity_notrot *= self.R / self.topo

        # update spherical harmonic model as well
        self.phase_velocity_ylm_notrot = \
            self.sh.analys(self.phase_velocity_notrot.copy())

        if self.group_velocity is not None:
            self.group_velocity_notrot *= self.R / self.topo
            self.group_velocity_ylm_notrot = \
                self.sh.analys(self.group_velocity_notrot.copy())

    def apply_ellipticity_correction(self, flattening=0.00589):
        f = flattening
        b = (1. - f) ** (2. / 3.) * self.R
        e = (2 * f - f ** 2) ** 0.5

        # generate mesh with theta and phi
        tm, pm = np.meshgrid(self.theta, self.phi)
        # see Yomogida 1985
        r_theta = b / (1. - e ** 2 * np.sin(tm) ** 2) ** 0.5

        # waves are faster where the radius is smaller:
        self.phase_velocity_notrot *= self.R / r_theta

        # update spherical harmonic model as well
        self.phase_velocity_ylm_notrot = \
            self.sh.analys(self.phase_velocity_notrot.copy())

        if self.group_velocity is not None:
            self.group_velocity_notrot *= self.R / r_theta
            self.group_velocity_ylm_notrot = \
                self.sh.analys(self.group_velocity_notrot.copy())

    def set_source_receiver(self, src_longitude, src_latitude, rec_longitude,
                            rec_latitude):
        '''
        rotates the velocity model such that the source and receiver are on
        the equator with the source at longitude zero and the azimuth towards
        the station is pi/2
        '''

        # transform to radians
        lat1 = math.radians(src_latitude)
        long1 = math.radians(src_longitude)
        lat2 = math.radians(rec_latitude)
        long2 = math.radians(rec_longitude)

        if long1 < 0:
            long1 += 2 * math.pi

        if long2 < 0:
            long2 += 2 * math.pi

        long_diff = long2 - long1

        # great circle distance (stable formula)
        gd = math.atan2(
                math.sqrt((
                    math.cos(lat2) * math.sin(long_diff)) ** 2 +
                    (math.cos(lat1) * math.sin(lat2) - math.sin(lat1) *
                        math.cos(lat2) * math.cos(long_diff)) ** 2),
                math.sin(lat1) * math.sin(lat2) +
                math.cos(lat1) * math.cos(lat2) * math.cos(long_diff))

        # compute azimuth first (stable formula)
        az = math.atan2(math.sin(long_diff),
                        math.cos(lat1) * math.tan(lat2) -
                        math.sin(lat1) * math.cos(long_diff))

        beta = az + math.pi
        c = lat1

        # now solve the triangle (not sure if formula for gamma is stable)
        gamma = math.acos(math.sin(beta) * math.cos(c))

        # stable version based on Napier's pentagon
        b = abs(math.atan2(math.sin(beta) * math.sin(c), math.cos(beta)))

        if long1 < long2:
            b *= -1

        if not long1 < long2 and gamma > math.pi / 2:
            b = 2 * math.pi - b
        elif long1 < long2 and gamma < math.pi / 2:
            b = 2 * math.pi - b

        # stable version based on Napier's pentagon
        a = math.acos(math.cos(c) * math.cos(b))

        if lat1 < 0:
            gamma *= -1
            b = math.pi - b
            a = math.pi - a

        self.rotangleZ1 = math.pi / 2. - (long1 - b)
        self.rotangleY = - gamma + math.pi
        self.rotangleZ2 = a - math.pi / 2.
        self.great_circle_distance = gd

        self.lat1 = lat1
        self.long1 = long1
        self.lat2 = lat2
        self.long2 = long2

        ylm = self.phase_velocity_ylm_notrot.copy()
        ylm = shtns.sht.Zrotate(self.sh, ylm, self.rotangleZ1)
        ylm = shtns.sht.Yrotate(self.sh, ylm, self.rotangleY)
        ylm = shtns.sht.Zrotate(self.sh, ylm, self.rotangleZ2)
        self.phase_velocity_ylm = ylm

        self.phase_velocity = self.sh.synth(ylm.copy())

        if self.group_velocity is not None:
            ylm = self.group_velocity_ylm_notrot.copy()
            ylm = shtns.sht.Zrotate(self.sh, ylm, self.rotangleZ1)
            ylm = shtns.sht.Yrotate(self.sh, ylm, self.rotangleY)
            ylm = shtns.sht.Zrotate(self.sh, ylm, self.rotangleZ2)
            self.group_velocity_ylm = ylm

            self.group_velocity = self.sh.synth(ylm.copy())

    def compute_travel_time_great_circle(self, orbit=1, epsrel=1e-5,
                                         which='group'):

        if which == 'phase':
            def __kernel(phi):
                return self.R / self.get_phase_velocity(np.pi / 2, phi)

        elif which == 'group':
            def __kernel(phi):
                return self.R / self.get_group_velocity(np.pi / 2, phi)
        else:
            raise ValueError()

        if orbit % 2 == 1:
            dist = self.great_circle_distance + np.pi * (orbit - 1)
        else:
            dist = -(np.pi * orbit - self.great_circle_distance)

        tt = np.abs(integrate.quad(__kernel, 0., dist, epsrel=epsrel)[0])
        return tt

    def compute_travel_time_1D(self, orbit=1, which='group'):

        if orbit % 2 == 1:
            dist = self.great_circle_distance + np.pi * (orbit - 1)
        else:
            dist = np.pi * orbit - self.great_circle_distance

        if which == 'group':
            return dist * self.R / self.hmean_group_velocity
        elif which == 'phase':
            return dist * self.R / self.hmean_phase_velocity
        else:
            raise ValueError()

    def calc_tt_map(self, nstep_dist, nstep_baz, rec_longitude = 136., rec_latitude = 5.):

        dists = np.linspace(0.01, 0.99, nstep_dist, endpoint=True) * np.pi * self.R
        bazs = np.linspace(-360 / (nstep_baz),
                           360 * (nstep_baz + 1) / (nstep_baz),
                           nstep_baz, endpoint=True)
        tt = np.zeros((len(dists), len(bazs)))
        for idist, dist in enumerate(dists):
            for ibaz, baz in enumerate(bazs):
                src_latitude, src_longitude = shoot(rec_latitude, rec_longitude,
                                                    bearing_degree=baz,
                                                    distance_km=dist, radius_km=self.R)
                self.set_source_receiver(src_longitude, src_latitude,
                                         rec_longitude, rec_latitude)
                tt[idist, ibaz] = self.compute_travel_time_great_circle(1, which='group')

                # fig = self.plot_velocity(show=False)
                # fig.savefig(fname='map_rotated_%d_%d.png' % (idist, ibaz))
                # fig = self.plot_velocity(show=False, rotated=False)
                # fig.savefig(fname='map_nonrot_%d_%d.png' % (idist, ibaz))
        return bazs, dists, tt

    def plot_velocity(self, show=True, rotated=True, source=True,
                      receiver=True, ms=10, cmap="RdGy", orbits=(0, 2),
                      which='phase'):

        fig = plt.figure()
        ax = plt.gca()

        if rotated:
            if which == 'phase':
                model = self.phase_velocity.T
            elif which == 'group':
                model = self.group_velocity.T
            else:
                raise ValueError('Unknown velocity %s' % which)
        else:
            if which == 'phase':
                model = self.phase_velocity_notrot.T
            elif which == 'group':
                model = self.group_velocity_notrot.T
            else:
                raise ValueError('Unknown velocity %s' % which)

        label = '%s velocity / (m / s)' % which

        model = np.tile(model, orbits[1] - orbits[0])
        plt.imshow(model, extent=[360. * orbits[0], 360. * orbits[1],
                                  -90., 90.], cmap=cmap)

        try:
            if rotated:
                if source:
                    plt.plot(0., 0., 'ro', ms=ms, zorder=10)
                    plt.plot(math.degrees(self.great_circle_distance),
                             0., 'g^', ms=ms, zorder=10)

                    plt.plot(360., 0., 'ro', ms=ms, zorder=10)
                    plt.plot(720., 0., 'ro', ms=ms, zorder=10)

                if receiver:
                    plt.plot(math.degrees(self.great_circle_distance) + 360.,
                             0., 'g^', ms=ms, zorder=10)

            else:
                if source:
                    plt.plot(math.degrees(self.long1), math.degrees(self.lat1),
                             'ro', ms=ms, zorder=10)
                    plt.plot(math.degrees(self.long1) + 360.,
                             math.degrees(self.lat1), 'ro', ms=ms, zorder=10)
                    plt.plot(math.degrees(self.long1) + 720.,
                             math.degrees(self.lat1), 'ro', ms=ms, zorder=10)

                if receiver:
                    plt.plot(math.degrees(self.long2), math.degrees(self.lat2),
                             'g^', ms=ms, zorder=10)

                    plt.plot(math.degrees(self.long2) + 360.,
                             math.degrees(self.lat2), 'g^', ms=ms, zorder=10)
        except AttributeError:
            warnings.warn("Source and receiver not initialized, skipping " +
                          "while plotting.")

        ax.set_xticks(np.linspace(360. * orbits[0], 360. * orbits[1],
                                  (orbits[1] - orbits[0]) * 4 + 1))
        ax.set_yticks([-90., -45., 0., 45, 90.])
        plt.xlabel('longitude')
        plt.ylabel('latitude')

        plt.xlim(360. * orbits[0], 360. * orbits[1])
        plt.ylim(-90., 90.)

        plt.colorbar(label=label, orientation='horizontal')

        if show:
            plt.show()
        else:
            return fig
