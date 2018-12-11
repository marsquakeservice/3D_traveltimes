#!/usr/bin/env python
"""

"""
__author__ = "Simon Stähler"
__license__ = "none"

import numpy as np
from scipy.interpolate import interp2d
from h5py import File
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

colors1 = plt.cm.viridis(np.linspace(0., 1, 128))
colors2 = plt.cm.viridis_r(np.linspace(0, 1, 128))

# combine them and build a new colormap
colors = np.vstack((colors1, colors2))
mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

# hdf5_file = '/home/staehler/CloudStation/Vorträge/20180927 - InSight STM 3D reloaded/output_2373_3090/mantlecrust_03090.h5'
hdf5_file = '../MQSORT_TAY.h5'
#hdf5_file = '../MQSORT_allthick.h5'
type = 'rayleigh'
#type = 'love'
dt_all = []
labels = []

with File(hdf5_file, 'r') as f:
    grp_surface_waves = f['surface_waves']

    bazs = np.array(grp_surface_waves['backazimuths'])
    dists = np.array(grp_surface_waves['distances'])
    periods = np.array(grp_surface_waves['periods'])

    for iperiod in [6, 12, 18]:
        tt = np.array(grp_surface_waves['period_%02d' % iperiod]['tt_%s' % type])

        ipl = interp2d(x=bazs, y=dists, z=tt, kind='quintic')
        path_pick = os.path.join('..', 'picked_times_Martin', 'model_TAY',
                                 'tt%s_%03d.txt' % (type[0], periods[iperiod]))

        print(path_pick)
        data = np.loadtxt(path_pick)
        dist = data[:, 0]
        #baz = (data[:, 1] + 180) % 360
        #baz = (180 - data[:, 1]) % 360
        baz = data[:, 1]
        time = data[:, 2]

        baz = baz[time > 0]
        dist = dist[time > 0]
        time = time[time > 0]

        # baz = baz[time < 3000]
        # dist = dist[time < 3000]
        # time = time[time < 3000]

        time_pred = np.zeros(len(dist))
        time_pred_1D = np.zeros(len(dist))
        for irec in range(len(dist)):
            time_pred[irec] = ipl(x=baz[irec], y=dist[irec])

            for b in bazs:
                time_pred_1D[irec] += ipl(x=b, y=dist[irec])
            time_pred_1D[irec] /= len(bazs)

        # fudgefactor = np.median((time_pred / time)[time_pred-time < 100])
        # time_pred /= fudgefactor
        # print(fudgefactor)

        slowness = np.median(time / dist)
        # time_pred_1D = dist * slowness
        fig, ax = plt.subplots(1, 5, figsize=(14,3))
        fig.suptitle('Period %4.1f sec' % periods[iperiod])
        sc = ax[0].scatter(dist, time - time_pred_1D, s=2, c=baz, cmap=mymap)
        ax[0].set_title('picked vs 1D prediction')
        ax[0].set_xlabel('distance')
        ax[0].set_ylabel('traveltime shift')
        #plt.colorbar(sc)
        ax[0].set_ylim(-500, 500)

        #sc = ax[1].scatter(time, time_pred_1D, s=2, c=baz, cmap=mymap)
        sc = ax[1].scatter(dist, time, s=2, c=baz, cmap=mymap)
        ax[1].set_title('travel time vs distance')
        #ax[1].set_ylabel('1D predicted time')
        ax[1].set_xlabel('distance')
        ax[1].set_ylabel('measured time')

        sc = ax[2].scatter(abs(time - time_pred_1D),
                           abs(time - time_pred),
                           s=2, c=baz, cmap=mymap)
        ax[2].set_title('1D vs 3D prediction')
        ax[2].set_ylabel('3D prediction error (absolute)')
        ax[2].set_xlabel('1D prediction error (absolute)')
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax[2].text(x=10, y=150, s='1D better', bbox=props)
        ax[2].text(x=120, y=40, s='3D better', bbox=props)
        ax[2].plot((-200, 200), (-200, 200), 'r--')
        ax[2].plot((-200, 200), (0, 0), 'r--')

        sc = ax[3].scatter(dist, time - time_pred, s=2, c=baz, cmap=mymap)
        ax[3].set_title('picked vs 3D prediction')
        ax[3].set_xlabel('distance')
        ax[3].set_ylabel('traveltime shift')
        ax[3].set_ylim(-500, 500)
        plt.colorbar(mappable=sc, ax=ax[4])


        for a in ax[2:3]:
            a.set_xlim(0, 200)
            a.set_ylim(0, 200)

        dT_1D = time - time_pred_1D
        dT_3D = time - time_pred # / fudgefactor

        #fig, ax = plt.subplots(1, 1)
        ax[4].hist([dT_1D.flatten(), dT_3D.flatten()],
                   bins = np.arange(-112.5, 120, 25),
                   label=['1D', '3D'])
        ax[4].set_title('sigma 1D: %5.1fs\n sigma 3D: %5.1f' %
                        (np.std(dT_1D[abs(dT_1D) < 250].flatten()),
                         np.std(dT_3D[abs(dT_3D) < 250].flatten())))
        ax[4].set_xlabel('seconds')
        ax[4].legend()
        plt.tight_layout()
        fig.savefig('%s_correction_%05.1fs.png' % (type, periods[iperiod]), dpi=300)
        plt.show()

        # plt.hist(dT_1D.flatten(), bins=np.arange(-212.5, 220, 25))
        dt_all.append(dT_3D.flatten())
        #plt.hist(dt_all, label=labels)
        labels.append('%5.1f s' % periods[iperiod])
        #plt.savefig('%s_picks_%05.1fs.png' % (type, periods[iperiod]))
#plt.hist(dt_all, label=labels, bins=np.arange(-400.0, 401, 50))

plt.show()
