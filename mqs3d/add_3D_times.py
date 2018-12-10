from mqs3d.crustal_thickness_to_phase_velocity import CrustalThicknessToVelocity, \
    filter_model_shtns, read_crustal_thickness_h5, write_model_h5

from mqs3d.surface_wave_ray_tracing import SurfaceWaveRayTracer
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d
import os
from h5py import File
from obspy.geodetics import kilometer2degrees

periods = [  5.00000,
             5.94604,
             7.07107,
             8.40896,
             10.00000,
             11.89207,
             14.14214,
             16.81793,
             20.00000,
             23.78414,
             28.28427,
             33.63586,
             40.00000,
             47.56828,
             56.56854,
             67.27171,
             80.00000,
             95.13657,
             113.13708,
             134.54343,
             160.00000]
R_mars = 3389.5e3


def plot_tts(swrt, **kwargs):
    bazs, dists, tt = swrt.calc_tt_map(nstep_baz=32, nstep_dist=24)

    tt_antipope = np.mean(tt[-1, :])
    tt_corr = np.zeros_like(tt)
    for idist in range(0, len(dists)):
        tt_corr[idist, :] = tt[idist, :] - dists[idist] * (tt_antipope / dists[-1])

    #np.savez('test.npz', tt=tt, bazs=bazs, dists=dists)

    ipl = interp2d(x=bazs, y=dists, z=tt, kind='cubic')

    idist = int(12)
    bazs_ipl = np.linspace(bazs[0], bazs[-1], 100)
    tt_ipl = ipl(x=bazs_ipl, y=dists[idist])
    tt_ipl_corr = tt_ipl[:] - dists[idist] * (tt_antipope / dists[-1])
    plt.plot(bazs_ipl, tt_ipl_corr, **kwargs)
    plt.xlabel(s='backazimuth')
    plt.ylabel(s='$\Delta$T / seconds')


def calc_disp_map(model_file, periods, plot=False):
    work_dir_local = '/scratch/snx3000/staehler' #os.path.join(work_dir, model_name)
    #work_dir_local = '/tmp' #os.path.join(work_dir, model_name)
    plot_dir = 'plots' #os.path.join(work_dir, model_name)
    topo, moho, lat, lon = read_crustal_thickness_h5(model_file)
    lmax_in = int(len(lat) / 4) - 1
    thickness = filter_model_shtns(topo - moho, lmax=lmax_in, order=6)
    topo = filter_model_shtns(topo, lmax=lmax_in, order=6)
    lmax_out = lmax_in * 2
    write_model_h5(topo, lmax=lmax_out,
                   fname=model_file,
                   varpath='ylm_maps/surface_radius')
    write_model_h5(moho, lmax=lmax_out,
                   fname=model_file,
                   varpath='ylm_maps/moho_radius')

    for type in ['rayleigh', 'love']:
        ctvelo = CrustalThicknessToVelocity(model_file, nthick=10, overtone_number=0,
                                            mtype=type,
                                            fmin=1./max(periods), fmax=1./min(periods), intk=2,
                                            with_topo=True, path_out=work_dir_local)

        if plot:
            plot_dispersion(ctvelo, plot_dir, thickness, periods, type)
        for iperiod, p in enumerate(periods):
            v = ctvelo.group_velocity(p, thickness)
            v = filter_model_shtns(v, lmax=lmax_in)
            write_model_h5(v, lmax=lmax_out,
                           fname=model_file,
                           varpath='ylm_maps/tt_%s_group_%02d' % (type, iperiod))

            v = ctvelo.phase_velocity(p, thickness)
            v = filter_model_shtns(v, lmax=lmax_in)
            write_model_h5(v, lmax=lmax_out,
                           fname=model_file,
                           varpath='ylm_maps/tt_%s_phase_%02d' % (type, iperiod))
    return


def plot_dispersion(ctvelo, plot_dir, thickness, periods, type):
    figs = ctvelo.plot(show=False)
    for fig in figs:
        fig.set_size_inches(6, 6, forward=True)
        fig.tight_layout()
    figs[0].savefig(os.path.join(plot_dir, type + '_group_thick.pdf'))
    figs[1].savefig(os.path.join(plot_dir, type + '_phase_thick.pdf'))
    plt.close('all')
    figs = ctvelo.plot_dispersion_curves(show=False)
    for fig in figs:
        fig.set_size_inches(6, 6, forward=True)
        fig.tight_layout()
    figs[0].savefig(os.path.join(plot_dir, type + '_group_disp.pdf'))
    figs[1].savefig(os.path.join(plot_dir, type + '_phase_disp.pdf'))
    plt.close('all')
    figs = ctvelo.plot_velocity_bias(thickness, show=False)
    for fig in figs:
        fig.set_size_inches(6, 4, forward=True)
        fig.tight_layout()
    figs[0].savefig(os.path.join(plot_dir, type + '_group_avg.pdf'))
    figs[1].savefig(os.path.join(plot_dir, type + '_group_bias.pdf'))
    figs[2].savefig(os.path.join(plot_dir, type + '_phase_avg.pdf'))
    figs[3].savefig(os.path.join(plot_dir, type + '_phase_bias.pdf'))
    plt.close('all')

    for p in periods:
         figs = ctvelo.plot_phase_velocity_map(p, thickness, mode='both',
                                               relative=True, show=False)
         for fig in figs:
             fig.set_size_inches(10, 5, forward=True)
             fig.tight_layout()
         figs[0].savefig(os.path.join(plot_dir, type + '_group_map_%03ds.pdf' % p))
         figs[1].savefig(os.path.join(plot_dir, type + '_phase_map_%03ds.pdf' % p))
         plt.close('all')


def write_tt_to_file(fname, periods, bazs, dists, tts):
    with File(fname, 'r+') as f:
        # print('Writing to %s' % fname)
        grp = f.create_group('surface_waves')
        grp.create_dataset('backazimuths', data=bazs, dtype='f2')
        dists_deg = kilometer2degrees(dists, radius=3389.5e3)
        grp.create_dataset('distances', data=dists_deg, dtype='f4')
        grp.create_dataset('periods', data=periods, dtype='f2')
        for iperiod, period in enumerate(periods):
            grp_period = grp.create_group('period_%02d' % iperiod)
            grp_period.create_dataset('p_c', data=period)
            grp_period.create_dataset('f_c', data=1./period)
            for type in ['rayleigh', 'love']:
                grp_period.create_dataset('tt_%s' % type,
                                          data=tts[type][iperiod],
                                          chunks=tts[type][iperiod].shape,
                                          dtype='f2')


def add_3D_traveltimes(model_file, verbose=False, plot=False):
    model_name = os.path.splitext(os.path.split(model_file)[-1])[0]
    work_dir = os.path.join('models_out')
    calc_disp_map(model_file, periods, plot=plot)

    swrt = SurfaceWaveRayTracer(R_mars, lmax=32, nphi=90, ntheta=48, verbose=verbose)
    tts = {'rayleigh': [],
           'love': []}
    cmap = {'rayleigh': plt.get_cmap('plasma'),
            'love': plt.get_cmap('viridis')}

    for type in ['rayleigh', 'love']:
        for iperiod in range(0, len(periods)):
            swrt.load_velocity_model_h5(fname=model_file, type=type, iperiod=iperiod)
            if plot:
                plot_tts(swrt, label='%s, %3d sec' % (type, periods[iperiod]),
                         color=cmap[type](float(iperiod) / len(periods)), ls='solid')

            swrt.apply_ellipticity_correction()
            if plot:
                plot_tts(swrt, label='%s, elli %3d sec' % (type, periods[iperiod]),
                         color=cmap[type](float(iperiod) / len(periods)), ls='dotted')

            swrt.apply_topography_correction()
            if plot:
                plot_tts(swrt, label='%s, topo %3d sec' % (type, periods[iperiod]),
                         color=cmap[type](float(iperiod) / len(periods)), ls='dashed')

            bazs, dists, tt = swrt.calc_tt_map(nstep_baz=32, nstep_dist=24)
            tts[type].append(tt)

        if plot:
            plt.title('Corrections, %s' % type)
            plt.grid(axis='both')
            plt.legend()
            plt.savefig('plots/TTcorr_90deg_%s_%s.pdf' % (model_name, type))
        plt.close('all')
    write_tt_to_file(model_file, periods, bazs, dists, tts)


if __name__ == '__main__':

    model_name = '015000'
    model_file = 'mantlecrust_%s.h5' % model_name
    add_3D_traveltimes(model_file)

#  def test_interpolation():
#      R_mars = 3390e3
#      swrt = SurfaceWaveRayTracer(R_mars, 32, nphi=360, ntheta=180,
#                                  delta_phi=0.01)
#      swrt.load_velocity_model_ylm(
#          'test_010s_phase.ylm')
#      rec_longitude, rec_latitude = 136., 5.
#      for nstep in [90, 32]:
#          bazs = np.linspace(-360/(nstep), 360*(nstep+1) / (nstep), nstep, endpoint=True)
#
#          dist = np.pi * R_mars * 0.29
#          tt = np.zeros(bazs.shape)
#          for ibaz, baz in enumerate(bazs):
#              src_latitude, src_longitude = shoot(rec_latitude, rec_longitude, bearing_degree=baz,
#                                                  distance=dist, radius=R_mars)
#              swrt.set_source_receiver(src_longitude, src_latitude,
#                                       rec_longitude, rec_latitude)
#              tt[ibaz] = swrt.compute_travel_time_great_circle(1, which='group')*1e3
#              print('%9.1f, %5.1f, %5.1f, %5.1f, %5.1f' % (dist*1e-3, baz, tt[ibaz],
#                                                    src_latitude, src_longitude))
#          #plt.plot(bazs, tt)
#          plt.plot(bazs, tt, 'o')
#          if nstep == 32:
#              ipl = interp1d(bazs, tt, kind='cubic')
#              x = np.arange(0, 360)
#              plt.plot(x, ipl(x), 'r')
#     plt.show()
