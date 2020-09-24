#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 17:30:02 2020

@author: mikkolp2
"""

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

''' Plotting '''
slice_1_dim = 1
slice_2_dim = 2



''' f_MAP '''
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(elev=90, azim=-90)
ax.scatter3D(GP_model.X[:,slice_1_dim-1], GP_model.X[:,slice_2_dim-1], GP_model.f_MAP, c=GP_model.f_MAP, cmap='hsv');
plt.show()

''' f_MAP random Fourier approximation '''
f_approx = np.dot(h_sampler.phi_X.T,h_sampler.omega_MAP)
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(elev=90, azim=-90)
ax.scatter3D(GP_model.X[:,slice_1_dim-1], GP_model.X[:,slice_2_dim-1], f_approx, c=f_approx, cmap='hsv');
plt.show()



