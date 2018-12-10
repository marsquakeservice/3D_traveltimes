#!/usr/bin/env python
"""

"""
__author__ = "Simon Stähler"
__license__ = "none"

import matplotlib
matplotlib.use('Agg')
from mqs3d.add_1D_times import create_taup, add_bodywave_times
from mqs3d.add_3D_times import add_3D_traveltimes
from mqs3d.calc_crust import calc_crust

import glob

#files = glob.glob('Models/MQSORT3D/model4mvdDW_AK74700_*')
#files = glob.glob('Models/MQSORT3D/model4mvdTAY*')
files = glob.glob('Models/FN/*')
files.sort()
verbose=True

for ifile, mantle_file in enumerate(files[0:1]):
    #mantle_file = 'Models/TR/BR_K_1650km_80km_2707kgm-3_522kgm-3_AC_2174K.dat'
    #mantle_file = 'Models/SH/C2_model_GAMA5_V3_Tm2000_Tc2500_Rc1855_eta02180_Estar300.deck'
    # h5_file = 'tmp_AK.h5' # ''mantlecrust_%06d.h5' % ifile
    #h5_file = 'MQSORT_TAY.h5' # ''mantlecrust_%06d.h5' % ifile
    h5_file = 'tmp_FN_%d.h5' % ifile # ''mantlecrust_%06d.h5' % ifile
    #h5_file = 'tmp_TR.h5' # ''mantlecrust_%06d.h5' % ifile

    # Calculate the crust map
    calc_crust(mantlefile=mantle_file,
               fnam_out_model=h5_file,
               fnam_out_plot='test',
               modeltype='FN',
               verbose=verbose)

    # # Do 3D surface wave travel times first
    add_3D_traveltimes(model_file=h5_file)

    # Add 1D body wave travel times now
    # First create TauPy-File
    taup_file = create_taup(model_file=h5_file)

    # Now fill H5 file with travel times
    add_bodywave_times(hdf5_file=h5_file, npz_file=taup_file, verbose=verbose)