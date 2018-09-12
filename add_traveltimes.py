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

files = glob.glob('modelsMQS')
files.sort()

for mantle_file in files:
    #mantle_file = 'modelsMQS/BR_K_1650km_80km_2707kgm-3_522kgm-3_AC_2174K.dat'
    h5_file = 'mantlecrust_016000.h5'

    # Calculate the crust map
    calc_crust(mantlefile=mantle_file,
               fnam_out_model=h5_file, fnam_out_plot='test')

    # Do 3D surface wave travel times first
    add_3D_traveltimes(model_file=h5_file)

    # Add 1D body wave travel times now
    # First create TauPy-File
    taup_file = create_taup(model_file=h5_file)

    # Now fill H5 file with travel times
    add_bodywave_times(hdf5_file=h5_file, npz_file=taup_file)