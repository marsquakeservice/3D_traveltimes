#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
A surface wave ray tracing class.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2016
:license:
    None
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.optimize import brentq, brenth
import shtns
import math
import warnings


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))


class surface_wave_ray_tracer(object):
    def __init__(self, R, lmax, nphi=None, ntheta=None, delta_phi=1.):
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

        else:
            raise ValueError()

    def _generate_rotation_test_model(self, src_longitude, src_latitude,
                                       rec_longitude, rec_latitude):
        # transform to radians
        lat1 = math.radians(src_latitude)
        long1 = math.radians(src_longitude)
        lat2 = math.radians(rec_latitude)
        long2 = math.radians(rec_longitude)

        shape = (self.nphi, self.ntheta)
        self.phase_velocity = np.zeros(shape) + 1e3

        idtheta = (np.abs(self.phi - long1)).argmin()
        idphi = (np.abs(self.theta - (math.pi / 2. - lat1))).argmin()
        self.phase_velocity[idtheta, idphi] = 1e3 + 1. / math.cos(lat1)

        idtheta = (np.abs(self.phi - long2)).argmin()
        idphi = (np.abs(self.theta - (math.pi / 2. - lat2))).argmin()
        self.phase_velocity[idtheta, idphi] = 1e3 - 1. / math.cos(lat2)

        ylm = self.sh.analys(self.phase_velocity.copy())
        self.phase_velocity_ylm_notrot = ylm
        self.phase_velocity_notrot = self.sh.synth(ylm.copy())

    def _generate_focussing_test_model(self, longitude, latitude, c0=1e3,
                                       delta_c=1e2, w=0.3):
        # transform to radians
        lat1 = math.radians(latitude)
        long1 = math.radians(longitude)

        shape = (self.nphi, self.ntheta)
        theta, phi = np.meshgrid(self.theta, self.phi, indexing='ij')
        self.phase_velocity = c0 + \
            delta_c * np.exp(-((theta.T - (math.pi / 2. - lat1)) / w) ** 2) * \
            np.exp(-((phi.T - long1) / w) ** 2)

        ylm = self.sh.analys(self.phase_velocity.copy())

        self.phase_velocity_ylm_notrot = ylm.copy()
        self.phase_velocity_notrot = self.sh.synth(ylm.copy())
        self.phase_velocity_ylm = ylm.copy()
        self.phase_velocity = self.sh.synth(ylm.copy())

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
        self.phase_velocity_notrot_ylm = \
            self.sh.analys(self.phase_velocity_notrot.copy())

        if self.group_velocity is not None:
            self.group_velocity_notrot *= self.R / r_theta
            self.group_velocity_notrot_ylm = \
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

    def __ray_tracing_kernel_kinematic(self, phi, y):
        """
        Kinematic ray-tracing equations (16.185 and 16.186) from Dahlen and
        Tromp 1998:
        dtheta/dphi = sin(theta) * cot(zeta)
        dzeta/dphi  = - cos(theta) + sin(theta) * dln(c)/dtheta
                      - cot(zeta) * dln(c)/dphi
        where c = w/k is the phase velocity
        """
        # assign some variables for convenience of notation
        theta = y[0]
        zeta = y[1]

        # cacluate the gradient
        dc_dtheta, dc_dphi = self.get_grad_phase_velocity(theta, phi)
        c = self.get_phase_velocity(theta, phi)
        cg = self.get_group_velocity(theta, phi)

        # kinematic ray-tracing
        dtheta_dphi = math.sin(theta) * math.cos(zeta) / math.sin(zeta)
        dzeta_dphi = -math.cos(theta) + math.sin(theta) * dc_dtheta / c - \
            math.cos(zeta) / math.sin(zeta) * dc_dphi / c
        dtg_dphi = math.sin(theta) / (math.sin(zeta) * cg)

        return [dtheta_dphi, dzeta_dphi, dtg_dphi]

    def __ray_tracing_kernel_kinematic_t(self, t, y):
        """
        Kinematic ray-tracing equations (16.182 and 16.184) from Dahlen and
        Tromp 1998, with the generarting parameter sigma replaces by
        t = sigma * omega * R / c ** 2 and k replaced by c = oemga / k
        """
        # assign some variables for convenience of notation
        theta = y[0]
        phi = y[1]
        zeta = y[2]

        # cacluate the gradient
        dc_dtheta, dc_dphi = self.get_grad_phase_velocity(theta, phi)
        c = self.get_phase_velocity(theta, phi)

        # kinematic ray-tracing
        dtheta_dt = c / self.R * math.cos(zeta)
        dphi_dt = c / self.R * math.sin(zeta) / math.sin(theta)
        dzeta_dt = math.sin(zeta) * dc_dtheta / self.R - \
                math.cos(zeta) / math.sin(theta)* dc_dphi / self.R - \
                math.cos(theta) / math.sin(theta) * math.sin(zeta) * c / self.R

        return [dtheta_dt, dphi_dt, dzeta_dt]

    def shoot_ray_t(self, shootingangle, tmax, rtol=1e-12, nsteps=10000,
                    dt=1.):
        """
        integrates the ray-tracing equations from the source to  a given
        shooting angle for time tmax
        """
        r = integrate.ode(self.__ray_tracing_kernel_kinematic_t)
        r.set_integrator('dopri5', nsteps=nsteps, rtol=rtol)

        n_steps = int(math.ceil(tmax / dt))

        t = np.linspace(0., tmax, n_steps + 1)

        # create vectors to store the ray
        theta = np.zeros(n_steps + 1)
        phi = np.zeros(n_steps + 1)
        zeta = np.zeros(n_steps + 1)

        # set the initial conditions
        theta[0] = np.pi / 2
        zeta[0] = shootingangle

        # initialize integrator
        r.set_initial_value([theta[0], phi[0], zeta[0]], 0.)

        # integrate the ODEs across each phi_steps step
        for i in np.arange(n_steps):
            r.integrate(t[i + 1])

            if not r.successful():
                raise RuntimeError('Integration Error')

            theta[i + 1], phi[i + 1], zeta[i + 1] = r.y

        return t, theta, phi, zeta

    def shoot_ray(self, shootingangle, orbit=1, return_path=False, rtol=1e-12,
                  nsteps=10000):
        """
        integrates the ray-tracing equations from source to receiver longitude
        for a given shooting angle
        """
        if orbit % 2 == 1:
            dist = self.great_circle_distance + np.pi * (orbit - 1)
        else:
            dist = np.pi * orbit - self.great_circle_distance

        # 'dopri5': explicit runge-kutta method of order (4)5 due to Dormand &
        # Prince
        r = integrate.ode(self.__ray_tracing_kernel_kinematic)
        r.set_integrator('dopri5', nsteps=nsteps, rtol=rtol)

        if return_path:
            n_steps = int(math.ceil(dist / self.delta_phi))
        else:
            n_steps = 1

        if orbit % 2 == 1:
            phi = np.linspace(0., dist, n_steps + 1)
        else:
            phi = np.linspace(0., -dist, n_steps + 1)

        # create vectors to store the ray
        theta = np.zeros(n_steps + 1)
        zeta = np.zeros(n_steps + 1)

        # set the initial conditions
        theta[0] = np.pi / 2
        zeta[0] = shootingangle

        # initialize integrator
        r.set_initial_value([theta[0], zeta[0], 0], 0.)

        # integrate the ODEs across each phi_steps step
        for i in np.arange(n_steps):
            r.integrate(phi[i + 1])

            if not r.successful():
                raise RuntimeError('Integration Error')

            theta[i + 1] = r.y[0]
            zeta[i + 1] = r.y[1]

        group_arrival = np.abs(r.y[2]) * self.R

        if return_path:
            return np.pi / 2. - theta[-1], theta, phi, group_arrival
        else:
            return np.pi / 2 - theta[-1]

    def find_rays(self, orbit=1, max_azimuth=45., nshots=50, rtol1=1e-5,
                  rtol2=1e-12, verbose=False):
        """
        shoots rays for nshots take of angles with azimuths from 90 -
        max_azimuth to 90 + max_azimuth degrees to the receiver longitude. If
        for two consecutive shots the arrival latitude changes, uses Brent's
        method to find the ray.
        """

        max_azimuth_rad = np.deg2rad(max_azimuth)
        min_shootingangle = np.pi / 2 - max_azimuth_rad
        max_shootingangle = np.pi / 2 + max_azimuth_rad

        if orbit % 2 == 0:
            min_shootingangle += np.pi
            max_shootingangle += np.pi

        shootingangles = np.linspace(min_shootingangle, max_shootingangle,
                                     nshots)

        rays = []
        try:
            theta1 = self.shoot_ray(shootingangles[0], orbit=orbit, rtol=rtol1)
        except:
            theta1 = 0.

        for i in np.arange(nshots-1):
            try:
                theta2 = self.shoot_ray(shootingangles[i+1], orbit=orbit,
                                        rtol=rtol1)
            except:
                theta2 = 0.

            if theta1 * theta2 < 0:
                rtmp, info = brentq(
                    self.shoot_ray, shootingangles[i], shootingangles[i+1],
                    args=(orbit, False, rtol2), full_output=True, disp=True)
                if verbose:
                    print(info)
                rays.append(rtmp)

            theta1 = theta2

        return rays

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
            if which == 'phase':
                model = self.phase_velocity_notrot.T
            elif which == 'group':
                model = self.group_velocity_notrot.T

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

        # ax.set_xticks([0., 90., 180., 270., 360.])
        #ax.set_xticks(np.linspace(0., 720, 9))
        ax.set_xticks(np.linspace(360. * orbits[0], 360. * orbits[1],
                                  (orbits[1] - orbits[0]) * 4 + 1))
        ax.set_yticks([-90., -45., 0., 45, 90.])
        plt.xlabel('longitude')
        plt.ylabel('latitude')

        plt.xlim(360. * orbits[0], 360. * orbits[1])

        # plt.xlim(0., 360.)
        # plt.xlim(0., 720.)
        # plt.xlim(0., 360. + np.rad2deg(self.great_circle_distance))
        plt.ylim(-90., 90.)

        plt.colorbar(label=label, orientation='horizontal')

        if show:
            plt.show()
        else:
            return fig


if __name__ == "__main__":

    R_mars = 3390e3

    swrt = surface_wave_ray_tracer(R_mars, 32, nphi=360, ntheta=180,
                                   delta_phi=0.01)

    orbit = 4
    #src_longitude, src_latitude = 0., -10.
    #rec_longitude, rec_latitude = 136., 0.
    #src_longitude, src_latitude = 0., 0.
    #rec_longitude, rec_latitude = 180., 0.

    # amp decrease
    src_longitude, src_latitude = 0., -15.
    rec_longitude, rec_latitude = 136., 5.

    swrt.load_velocity_model_ylm(
        'test_010s_phase.ylm')
    # swrt.apply_ellipticity_correction()


    swrt.set_source_receiver(src_longitude, src_latitude,
                             rec_longitude, rec_latitude)
    #swrt.plot_phase_velocity()

    rtol1 = 1e-7
    rtol2 = 1e-7
    rays = swrt.find_rays(orbit=orbit, max_azimuth=60., nshots=100, rtol1=rtol1,
                          rtol2=rtol2, verbose=True)


    #swrt.plot_phase_velocity(show=False, rotated=False)
    swrt.plot_velocity(show=False)

    plt.show()

    swrt.plot_velocity(show=False)
    plt.xlim(0., 180.)
    plt.ylim(-45., 45.)

    plt.show()
