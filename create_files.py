#!/usr/bin/env python
"""

"""
__author__ = "Andrea Saxer, Simon Stähler"
__license__ = "none"

import matplotlib
matplotlib.use('Agg')
from mqs3d.add_1D_times import create_taup, add_bodywave_times
from mqs3d.add_3D_times import add_3D_traveltimes
from mqs3d.calc_crust import calc_crust
from multiprocessing import Pool
import sys, os


def add_mult(file):

    with open('MVD/model_names_AK.txt', 'r') as f:  # changed to work with text file (12.09.18)
        ifile = f.read().splitlines().index(file[4:])

    h5_file = 'output/mantlecrust_%05d.h5' % ifile

    # Calculate the crust map
    print('\n=====     %05d   -   1/3  Crust map & 1D Mantle     =====' % ifile)
    calc_crust(mantlefile=file,
                fnam_out_model=h5_file,
                fnam_out_plot=h5_file[:-3],
                gravfile='Data/gmm3_120_sha.tab',
                topofile='Data/MarsTopo719.shape',
                densityfile='Data/dichotomy_359.sh')


    # Do 3D surface wave travel times first
    print('\n=====     %05d   -   2/3  3D surface wave travel times     =====' % ifile)
    add_3D_traveltimes(model_file=h5_file)


    # Add 1D body wave travel times now
    # First create TauPy-File
    taup_file = create_taup(model_file=h5_file)

    # Now fill H5 file with travel times
    print('\n====     %05d   -   3/3  1D body waves travel times     =====' % ifile)
    add_bodywave_times(hdf5_file=h5_file, npz_file=taup_file)


# execute

with open('MVD/model_names_all.txt', 'r') as f:
    files = f.read().splitlines()
files = ['MVD/' + file for file in files]

#for file in files:
#    add_mult(file)
with Pool(10) as p:
    p.map(add_mult, files)
