import numpy as np
from obspy.taup import TauPyModel
import h5py
# from tqdm import tqdm
import os


def create_taup(model_file, taup_path ='taup_files'):
    from obspy.taup.taup_create import build_taup_model
    with h5py.File(model_file, 'r') as f:
        mantle = f['mantle']
        index_moho = int(mantle['moho_layer'].value)
        index_cmb = int(mantle['cmb_layer'].value)
        fnam_out = os.path.splitext(model_file)[0] + '.nd'
        nlayer = len(mantle['vp'])
        with open(fnam_out, 'w') as fid:
            for i in np.arange(nlayer - 1, 0, -1):
                if i == index_moho - 1:
                    fid.write('mantle\n')
                if i == index_cmb:
                    fid.write('outer-core\n')
                fid.write('%7.2f %6.4f %6.4f %6.4f\n' %
                          (1e-3 * (mantle['radius'][-1] - mantle['radius'][i]),
                           1e-3 * mantle['vp'][i],
                           1e-3 * mantle['vs'][i],
                           1e-3 * mantle['rho'][i] ))
            fid.write('%7.2f %6.4f %6.4f %6.4f\n' %
                      (3389.3,
                       1e-3 * mantle['vp'][0],
                       1e-3 * mantle['vs'][0],
                       1e-3 * mantle['rho'][0] ))

            # Create an artificial inner core, so that TauPy
            # works correctly for core phases
            fid.write('inner-core\n')
            fid.write('%7.2f %6.4f %6.4f %6.4f\n' %
                      (3389.3,
                       1e-3 * mantle['vp'][0],
                       3.0,
                       1e-3 * mantle['rho'][0] ))
            fid.write('%7.2f %6.4f %6.4f %6.4f\n' %
                      (3389.5,
                       1e-3 * mantle['vp'][0],
                       3.0,
                       1e-3 * mantle['rho'][0] ))
        model_name = os.path.splitext(os.path.split(fnam_out)[-1])[0]
        taup_name = os.path.join(taup_path, model_name + '.npz')
        if not os.path.exists(taup_name):
            build_taup_model(fnam_out, output_folder=taup_path)
        return taup_name


def add_bodywave_times(hdf5_file, npz_file):

    model = TauPyModel(npz_file)

    depths = [0.00, 0.2, 0.4, 0.6, 0.8, 1.00, 1.5, 2.00, 3.00, 4.00, 5.00, 10.00, 15.00, 20.00,
              25.00, 30.00, 33.00, 35.00, 40.00, 45.00, 50.00, 55.00, 60.00,
              70.00, 80.00, 90.00, 100.00, 120.00, 140.00, 160.00, 180.00, 200.00, 220.00,
              240.00, 260.00, 280.00, 300.00, 350.00, 400.00, 450.00, 500.00, 550.0, 600.00,
              700.00, 800.00, 900.00, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
    dists = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25,
             0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90,
             1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90,
             2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90,
             3.00, 3.10, 3.20, 3.30, 3.40, 3.50, 3.60, 3.70, 3.80, 3.90,
             4.00, 4.10, 4.20, 4.30, 4.40, 4.50, 4.60, 4.70, 4.80, 4.90,
             5.00, 5.25, 5.50, 5.75,
             6.00, 6.25, 6.50, 6.75,
             7.00, 7.25, 7.50, 7.75,
             8.00, 8.25, 8.50, 8.75,
             9.00, 9.25, 9.50, 9.75,
             10.00, 10.25, 10.50, 10.75, 
             11.00, 11.25, 11.50, 11.75, 
             12.00, 12.25, 12.50, 12.75, 
             13.00, 13.25, 13.50, 13.75, 
             14.00, 14.25, 14.50, 14.75, 
             15.00, 15.25, 15.50, 15.75, 
             16.00, 16.25, 16.50, 16.75, 
             17.00, 17.25, 17.50, 17.75, 
             18.00, 18.25, 18.50, 18.75, 
             19.00, 19.25, 19.50, 19.75, 
             20.00, 21.00, 22.00, 23.00, 24.00, 25.00, 26.00, 27.00, 28.00, 29.00,
             30.00, 31.00, 32.00, 33.00, 34.00, 35.00, 36.00, 37.00, 38.00, 39.00,
             40.00, 41.00, 42.00, 43.00, 44.00, 45.00, 46.00, 47.00, 48.00, 49.00,
             50.00, 51.00, 52.00, 53.00, 54.00, 55.00, 56.00, 57.00, 58.00, 59.00,
             60.00, 61.00, 62.00, 63.00, 64.00, 65.00, 66.00, 67.00, 68.00, 69.00,
             70.00, 71.00, 72.00, 73.00, 74.00, 75.00, 76.00, 77.00, 78.00, 79.00,
             80.00, 81.00, 82.00, 83.00, 84.00, 85.00, 86.00, 87.00, 88.00, 89.00,
             90.00, 91.00, 92.00, 93.00, 94.00, 95.00, 96.00, 97.00, 98.00, 99.00,
             100.00, 105.00, 110.00, 115.00, 120.00, 125.00, 130.00, 135.00, 140.00, 145.00,
             150.00, 155.00, 160.00, 165.00, 170.00, 175.00, 179.50]

    phase_names = ['p', 's', 'pP', 'sP', 'pS', 'sS', 'P', 'S', 'PP', 'SS', 'PPP', 'SSS', 'SP', 'PS',
                   'Pn', 'Sn', 'Pg', 'Sg', 'PcP', 'ScSScS', 'PmP', 'SmS', 
                   'ScS', 'PKKP', 'SKKS', 'Pdiff', 'Sdiff', 'PKP', 'SKS', 'P1', 'S1']


    n_phase = {}
    for counter, phase in enumerate(phase_names):
        n_phase[phase] = counter

    phase_names_P1 = ['p', 'Pg', 'Pn', 'P', 'PP', 'PKP']
    phase_names_S1 = ['s', 'Sg', 'Sn', 'S', 'SS', 'SKS']

    times = np.ones((len(depths), len(dists), len(phase_names))) * -1
    inc_angles = np.zeros_like(times)
    slownesses = np.zeros_like(inc_angles)
    mod = model.model
    vp_crust = mod.s_mod.v_mod.evaluate_above(mod.moho_depth/2., 'p')
    for x, depth_i in enumerate(depths):
        for y, dist_i in enumerate(dists):
            try:
                arrs = model.get_travel_times(source_depth_in_km=depth_i,
                                              distance_in_degree=dist_i,
                                              phase_list=phase_names[:-2])
            except:
                pass
            else:
                # Fill Pg travel time with phase of group velocity of mid crust
                times[x, y, n_phase['Pg']] = np.deg2rad(dist_i) * \
                                             mod.radius_of_planet / vp_crust
                for arr in np.flip(arrs, 0):
                    times[x, y, n_phase[arr.phase.name]] = arr.time
                    inc_angles[x, y, n_phase[arr.phase.name]] = arr.incident_angle
                    slownesses[x, y, n_phase[arr.phase.name]] = arr.ray_param
                    if arr.phase.name in phase_names_P1:
                        times[x, y, n_phase['P1']] = arr.time
                        inc_angles[x, y, n_phase['P1']] = arr.incident_angle
                        slownesses[x, y, n_phase['P1']] = arr.ray_param
                    if arr.phase.name in phase_names_S1:
                        times[x, y, n_phase['S1']] = arr.time
                        inc_angles[x, y, n_phase['S1']] = arr.incident_angle
                        slownesses[x, y, n_phase['S1']] = arr.ray_param

    with h5py.File(hdf5_file, 'r+') as f:
        grp_body_waves = f.create_group('body_waves')
        grp_body_waves.create_dataset('distances', data=dists)
        grp_body_waves.create_dataset('depths', data=depths)
        grp_body_waves.create_dataset('phase_names', data=[n.encode("ascii", "ignore") for n in phase_names])
        grp_body_waves.create_dataset('times', data=times, dtype='f2')
        grp_body_waves.create_dataset('inc_angles', data=inc_angles, dtype='f2')
        grp_body_waves.create_dataset('slowness', data=slownesses, dtype='f2')


if __name__ == '__main__':
    # test
    #create_taup('mantlecrust_015000.h5')
    add_bodywave_times('mantlecrustbody_test.h5', 'taup_files/mantlecrust_016000.npz')
