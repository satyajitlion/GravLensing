# imports all the modules needed

import math
import numpy as np
import matplotlib.pyplot as plt
import pygravlens as gl
from astropy.cosmology import Planck18 as cosmo

# constant vals

# number of mock lenses
num_mock = 30

# generates random einstein radii 
EinsArr = np.random.uniform(1.0, 1.5, num_mock) 

# from a radius of 0 to 1 in polar coords
r = np.sqrt(np.random.uniform(0.0, 1.0, num_mock))

theta_src = np.random.uniform(0, 2*math.pi, num_mock)

# random source positions
betaOne = [];
betaTwo = [];

for i in range(num_mock):
    # x = rcos theta
    # y = rsin theta
    betaOne.append(r[i]*np.cos(theta_src[i]))
    betaTwo.append(r[i]*np.sin(theta_src[i]))

# randomized shear vals b/w 0 and 0.1 and ellipticity vals between 0 and 0.5
shear_vals = np.random.uniform(0, 0.1, num_mock)
ellip_vals = np.random.uniform(0, 0.5, num_mock)

# random theta vals

theta_e = np.random.uniform(0, 2*math.pi, num_mock)
theta_g = np.random.uniform(0, 2*math.pi, num_mock)

# ec, es; gc, gs
'''
ec = ellip*cos(2*theta_e)
es = ellip*sin(2*theta_e)
gc = gamma*cos(2*theta_g)
gs = gamma*sin(2*theta_g)
'''
ec = [];
es = [];
gc = [];
gs = [];

for i in range(num_mock):
    ec.append(ellip_vals[i]*np.cos(2*theta_e[i]))
    es.append(ellip_vals[i]*np.sin(2*theta_e[i]))
    gc.append(shear_vals[i]*np.cos(2*theta_g[i]))
    gs.append(shear_vals[i]*np.sin(2*theta_g[i]))

# accounting for red-shift:
zlens = np.random.uniform(0.2, 0.5, num_mock);
zsrc = np.random.uniform(1.0, 3.0, num_mock);

Dlens = [];
Dsrc = [];

for i in range(num_mock):
    Dlens.append(cosmo.comoving_distance(zlens[i]));
    Dsrc.append(cosmo.comoving_distance(zsrc[i]));

xtmp_elpow = np.random.uniform(low=-2,high=2,size=(1000,2))

