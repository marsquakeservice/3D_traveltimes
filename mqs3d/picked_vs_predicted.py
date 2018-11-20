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

hdf5_file = '/home/staehler/CloudStation/Vorträge/20180927 - InSight STM 3D reloaded/output_2373_3090/mantlecrust_03090.h5'
type = 'love'
dt_all = []
labels = []

with File(hdf5_file, 'r') as f:
    grp_surface_waves = f['surface_waves']

    bazs = np.array(grp_surface_waves['backazimuths'])
    dists = np.array(grp_surface_waves['distances'])
    periods = np.array(grp_surface_waves['periods'])

    for iperiod in [0, 6, 12, 18]:
        tt = np.array(grp_surface_waves['period_%02d' % iperiod]['tt_%s' % type])
        ipl = interp2d(x=bazs, y=dists, z=tt, kind='quintic')
        path_pick = os.path.join('..', 'picked_times_Martin',
                                 'tt%s_%03d.txt' % (type[0], periods[iperiod]))

        print(path_pick)
        data = np.loadtxt(path_pick)
        dist = data[:, 0]
        baz = (data[:, 1] + 180) % 360
        time = data[:, 2]

        baz = baz[time > 0]
        dist = dist[time > 0]
        time = time[time > 0]

        baz = baz[time < 3000]
        dist = dist[time < 3000]
        time = time[time < 3000]

        time_pred = np.zeros(len(dist))
        for irec in range(len(dist)):
            time_pred[irec] = ipl(x=baz[irec], y=dist[irec])
        #time_pred = np.asarray(time_pred)

        fudgefactor = np.median(time_pred / time)
        time_pred /= fudgefactor
        print(fudgefactor)

        slowness = np.median(time/ dist)
        time_pred_1D = dist * slowness
        fig, ax = plt.subplots(1, 4, figsize=(14,4))
        sc = ax[0].scatter(dist, time - time_pred_1D, s=2, c=baz, cmap=mymap)
        ax[0].set_xlabel('distance')
        ax[0].set_ylabel('traveltime shift')
        #plt.colorbar(sc)
        ax[0].set_ylim(-500, 500)

        sc = ax[1].scatter(time, time_pred_1D, s=2, c=baz, cmap=mymap)
        ax[1].set_ylabel('1D predicted time')
        ax[1].set_xlabel('measured time')

        sc = ax[2].scatter(time, time_pred, s=2, c=baz, cmap=mymap)
        ax[2].set_ylabel('3D predicted time')
        ax[2].set_xlabel('measured time')

        sc = ax[3].scatter(dist, time - time_pred, s=2, c=baz, cmap=mymap)
        ax[3].set_xlabel('distance')
        ax[3].set_ylabel('traveltime shift')
        #plt.colorbar(sc)
        ax[3].set_ylim(-500, 500)
        plt.tight_layout()
        plt.savefig('%s_picks_%5.1fs.png' % (type, periods[iperiod]))

        plt.show()

        for a in ax[1:3]:
            a.set_xlim(0, 3000)
            a.set_ylim(0, 3000)

        dT = time - time_pred
        dT_1D = time - time_pred_1D # / fudgefactor

        # plt.hist([dT.flatten(), dT_1D.flatten() / 4],
        #          bins = np.arange(-212.5, 220, 25),
        #          label=['difference to 1D', 'corrected'])
        # plt.xlabel('seconds')
        # plt.legend()
        # plt.savefig('correction_%5.1fs.png' % periods[iperiod])
        # plt.show()

        # plt.hist(dT_1D.flatten(), bins=np.arange(-212.5, 220, 25))
        dt_all.append(dT_1D.flatten())
        #plt.hist(dt_all, label=labels)
        labels.append('%5.1f s' % periods[iperiod])
#plt.hist(dt_all, label=labels, bins=np.arange(-400.0, 401, 50))

plt.show()
