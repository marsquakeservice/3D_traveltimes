#!/usr/bin/env python3

# coding: utf-8

"""
pyCrust_Mars

Create a crustal thickness map of Mars from gravity and topography.

This script generates two different crustal thickness maps. The first assumes
that the density of both the crust and mantle are constant, whereas the second
includes the effect of different densities on either side of the dichotomy
boundary. The average crustal thickness is iterated in order to obtain
a specified minimum crustal thickness.
"""
import numpy as np
import pyshtools
import pyCrust
from pyCrust.Hydrostatic import HydrostaticShapeLith
import glob
import h5py


def calc_crust(mantlefile,    # filename for mantle profile input
               fnam_out_model,      # Output file name for thickness map
               gravfile=pyCrust.gravfile,
               topofile=pyCrust.topofile,
               densityfile=pyCrust.densityfile,
               fnam_out_plot=None,      # Output file name for plot of map
               t0=1.e3,       # minimum crustal thickness
               d_lith=300.e3  # Lithosphere thickness
               ):

    lmax_calc = 90
    lmax = lmax_calc * 4
    grav_constant = 6.674e-11

    potcoefs, lmaxp, header = pyshtools.shio.shread(gravfile, header=True,
                                                    lmax=lmax)
    potential = pyshtools.SHCoeffs.from_array(potcoefs)
    potential.r_ref = float(header[0]) * 1.e3
    potential.gm = float(header[1]) * 1.e9
    potential.mass = potential.gm / float(grav_constant)

    print('Gravity file = {:s}'.format(gravfile))
    print('Lmax of potential coefficients = {:d}'.format(lmaxp))
    print('Reference radius (km) = {:f}'.format(potential.r_ref / 1.e3))
    print('GM = {:e}\n'.format(potential.gm))

    topo = pyshtools.SHCoeffs.from_file(topofile, lmax=lmax)
    topo.r0 = topo.coeffs[0, 0, 0]

    print('Topography file = {:s}'.format(topofile))
    print('Lmax of topography coefficients = {:d}'.format(topo.lmax))
    print('Reference radius (km) = {:f}\n'.format(topo.r0 / 1.e3))

    density = pyshtools.SHCoeffs.from_file(densityfile, lmax=lmax)

    print('Lmax of density coefficients = {:d}\n'.format(density.lmax))

    lat_insight = 4.43
    lon_insight = 135.84

    filter = 1
    half = 50
    nmax = 7
    lmax_hydro = 15
    t0_sigma = 5.  # maximum difference between minimum crustal thickness
    omega = float(pyshtools.constant.omega_mars.to_value())

    # --- read 1D reference interior model ---

    print('=== Reading model {:s} ==='.format(mantlefile))

    with open(mantlefile, 'r') as f:
        lines = f.readlines()

    ncomments = 4  # Remove initial four lines in AxiSEM files
    nlines = len(lines)
    nlayer = nlines - ncomments
    radius = np.zeros(nlayer)
    rho = np.zeros(nlayer)
    vp = np.zeros(nlayer)
    vs = np.zeros(nlayer)
    qmu = np.zeros(nlayer)       # added qmu & qka, 03.09.2018
    qka = np.zeros(nlayer)
    lines = lines[::-1]

    for i in range(0, nlayer):
        data = lines[i].split()
        radius[i] = float(data[0])
        rho[i] = float(data[1])
        vp[i] = float(data[2])
        vs[i] = float(data[3])
        qmu[i] = float(data[4])
        qka[i] = float(data[5])

    if radius[nlayer-1] == 0:       # added for Rivoldini models, 11.09.2018
        radius = np.flip(radius, axis=0)
        rho = np.flip(rho, axis=0)
        vp = np.flip(vp, axis=0)
        vs = np.flip(vs, axis=0)
        qmu = np.flip(qmu, axis=0)
        qka = np.flip(qka, axis=0)

        crust_index = nlayer - 21  # fits for all Rivoldini models (qmu < 600)
    else:
        crust_index = nlayer - 6  # True for all Khan models


    vs[vs < 1] = 0  # added (for vs to be = 0 in the core), 04.09.2018

    # Calculate crustal density
    mass_crust = 0
    for i in range(crust_index, nlayer-1):
        vol_layer = (radius[i+1] - radius[i]) * 1e3 * 4 * np.pi
        mass_crust += rho[i] * vol_layer
    vol_crust = (radius[-1] - radius[crust_index]) * 1e3 * 4 * np.pi
    rho_c = mass_crust / vol_crust
    print('Crustal density: = {:8.1f}'.format(rho_c))

    r0_model = radius[nlayer-1]
    print('Surface radius of model (km) = {:8.1f}'.format(r0_model / 1.e3))

    # Find layer at bottom of lithosphere
    for i in range(0, nlayer):
        if radius[i] <= (r0_model - d_lith) and \
                radius[i+1] > (r0_model - d_lith):
            if radius[i] == (r0_model - d_lith):
                i_lith = i
            elif (r0_model - d_lith) - radius[i] <= \
                    radius[i+1] - (r0_model - d_lith):
                i_lith = i
            else:
                i_lith = i + 1
            break

    n = nlayer - 1
    rho[n] = 0.  # the density above the surface is zero
    rho_mantle = rho[crust_index-1]
    print('Mantle density (kg/m3) = {:8.1f}'.format(rho_mantle))

    print('Assumed depth of lithosphere (km) = {:6.1f}'.format(d_lith / 1.e3))
    print('Actual depth of lithosphere in discretized model (km) = {:6.1f}'
          .format((r0_model - radius[i_lith]) / 1.e3))

    # initial guess of average crustal thickness
    thickave = r0_model - radius[crust_index]
    print('Crustal thickness (km) = {:5.1f}'.format(thickave / 1e3))
    print('Moho layer: {:d}'.format(crust_index))

    # --- Compute gravity contribution from hydrostatic density interfaces ---

    hlm, clm_hydro, mass_model = HydrostaticShapeLith(radius, rho, i_lith,
                                                      potential, omega,
                                                      lmax_hydro,
                                                      finiteamplitude=False)

    print('Total mass of model (kg) = {:e}'.format(mass_model))
    print('% of J2 arising from beneath lithosphere = {:f}'
          .format(clm_hydro.coeffs[0, 2, 0]/potential.coeffs[0, 2, 0] * 100.))

    potential.coeffs[:, :lmax_hydro+1, :lmax_hydro+1] -= \
        clm_hydro.coeffs[:, :lmax_hydro+1, :lmax_hydro+1]

    # --- Constant density model ---
    print('-- Constant density model --\nrho_c = {:f}'.format(rho_c))

    tmin = 1.e9
    converged = False
    while not converged:
        # iterate to fit assumed minimum crustal thickness

        moho = pyCrust.pyMoho(potential, topo, lmax, rho_c, rho_mantle,
                              thickave, filter_type=filter, half=half,
                              lmax_calc=lmax_calc, nmax=nmax, quiet=True)
        thick = topo.pad(lmax) - moho.pad(lmax)
        print('Average crustal thickness (km) = {:6.2f}'.format(thickave /
                                                                1.e3))
        thick_insight = thick.expand(lat=lat_insight, lon=lon_insight)
        print('Crustal thickness at InSight landing sites (km) = {:6.2f}'
              .format(thick_insight / 1.e3))

        thick_grid = thick.expand(grid='DH2')
        tmin = thick_grid.data.min()
        tmax = thick_grid.data.max()
        print('Minimum thickness (km) = {:6.2f}'.format(tmin / 1.e3))
        print('Maximum thickness (km) = {:6.2f}'.format(tmax / 1.e3))

        if tmin - t0 < - t0_sigma:
            thickave += t0 - tmin
        else:
            converged = True


    # Write Model to disk
    lats = np.arange(-87.5, 90., 5.)
    lons = np.arange(0, 360, 5)


    # Apply tapered anti-aliasing filter to SH before transformation
    lmax_filter = 36
    order = 2
    lvals = np.zeros_like(topo.coeffs)
    for i in range(0, lvals.shape[1]):
        for j in range(0, lvals.shape[2]):
            l = np.max([i, j])
            lvals[:, i, j] = np.exp(-2. * np.pi * l ** order /
                                    (2. * lmax_filter) ** order)

    topo.coeffs *= lvals
    lvals = np.zeros_like(moho.coeffs)
    for i in range(0, lvals.shape[1]):
        for j in range(0, lvals.shape[2]):
            l = np.max([i, j])
            lvals[:, i, j] = np.exp(-2. * np.pi * l ** order /
                                    (2. * lmax_filter) ** order)
    moho.coeffs *= lvals

    lats_grid, lons_grid = np.meshgrid(lats, lons)
    topo_grid = (topo).expand(lat=lats_grid, lon=lons_grid)
    moho_grid = (moho).expand(lat=lats_grid, lon=lons_grid)

    if fnam_out_plot:
        # Plot full resolution thickness grid
        fnam = fnam_out_plot + '_thickness_full.png'
        thick_grid /= 1e3
        thick_grid.plot(show=False, fname=fnam,
                        colorbar=True,
                        vmin=0.0, vmax=120.)

        fnam = fnam_out_plot + '_moho.png'
        moho_low = moho.expand(grid='DH2') / 1e3
        moho_low.plot(show=False,
                      fname=fnam,
                      colorbar=True,
                      vmin=3220.0,
                      vmax=3370.0)
        fnam = fnam_out_plot + '_topo.png'
        topo_low = topo.expand(grid='DH2') / 1e3
        topo_low.plot(show=False,
                      fname=fnam,
                      colorbar=True,
                      vmin=3360.0,
                      vmax=3410.0)
        fnam = fnam_out_plot + '_thickness.png'
        thick_low = (topo.pad(lmax) - moho.pad(lmax)).expand(grid='DH2') / 1e3
        thick_low.plot(show=False,
                       fname=fnam,
                       colorbar=True,
                       vmin=0.0,
                       vmax=120.)

    with h5py.File(fnam_out_model, 'w') as f:
        f.create_dataset('model_name', data=mantlefile)
        print('Writing to %s' % fnam_out_model)
        grp_crust = f.create_group('crust')
        grp_crust.create_dataset('moho', data=moho_grid.data)
        grp_crust.create_dataset('topo', data=topo_grid.data)
        grp_crust.create_dataset('latitudes', data=lats)
        grp_crust.create_dataset('longitudes', data=lons)

        # Correct the density in the uppermost layer
        rho[n] = rho[n-1]

        grp_mantle = f.create_group('mantle')
        grp_mantle.create_dataset('radius', data=radius)
        grp_mantle.create_dataset('rho', data=rho)
        grp_mantle.create_dataset('vp', data=vp)
        grp_mantle.create_dataset('vs', data=vs)
        grp_mantle.create_dataset('qmu', data=qmu)  # added qmu & qka, 03.09.2018
        grp_mantle.create_dataset('qka', data=qka)

        grp_status = f.create_group('status')
        grp_status.create_dataset('thickness_insight', data=thick_insight)
        grp_status.create_dataset('thickness_average_pycrust', data=thickave)
        grp_status.create_dataset('thickness_average_input',
                                  data=r0_model - radius[crust_index])
        grp_status.create_dataset('thickness_lithosphere', data=d_lith)
        grp_status.create_dataset('thickness_minimum', data=tmin)
        grp_status.create_dataset('thickness_maximum', data=tmax)
        grp_status.create_dataset('rho_crust', data=rho_c)
        grp_status.create_dataset('rho_mantle', data=rho_mantle)


def calc_crust_mult(fnam_in):
    print(fnam_in)
    # ifile = int(fnam_in[-9:-4])
    ifile = fnam_in[8:-4]           # changed to contain whole name, 11.09.2018
    calc_crust(mantlefile=fnam_in,
               # fnam_out_model='output/mantlecrust_%06d.h5' % ifile,
               fnam_out_model='output/mantlecrust_%s.h5' % ifile,
               gravfile='Data/gmm3_120_sha.tab',
               topofile='Data/MarsTopo719.shape',
               densityfile='Data/dichotomy_359.sh')


# ==== EXECUTE SCRIPT ====
if __name__ == "__main__":
    # if len(sys.argv) > 1:
    #     fnam_in = sys.argv[1]
    #     fnam_out = sys.argv[2]
    #     calc_crust(mantlefile=fnam_in, fnam_out_model=fnam_out)
    # else:
    dir_in = 'MVD_all/*???00.dat'  # changed to input all models
    files = glob.glob(dir_in)
    files.sort()
    for file in files:
        calc_crust_mult(file)
    with Pool(1) as p:
        p.map(calc_crust_mult, files)