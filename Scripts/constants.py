# imports all the modules needed

import math
import numpy as np
import matplotlib.pyplot as plt
import pygravlens as gl
from astropy.cosmology import Planck18 as cosmo

# constant vals

# number of mock lenses
num_mock = 10**4

# generates random einstein radii 
EinsArr = np.random.uniform(1.0, 1.5, num_mock) 

# from a radius of 0 to 1 in polar coords
r = np.sqrt(np.random.uniform(0.0, 1.0, num_mock))

# sources:
# theta_src = np.random.uniform(0, 2*math.pi, num_mock)

# images:

theta_im = np.random.uniform(0, 2*math.pi, num_mock)
im1 = [];
im2 = [];

# random source positions
betaOne = [];
betaTwo = [];

for i in range(num_mock):
    im1.append(r[i]*np.cos(theta_im[i]));
    im2.append(r[i]*np.sin(theta_im[i]));

'''
for i in range(num_mock):
    # x = rcos theta
    # y = rsin theta
    betaOne.append(r[i]*np.cos(theta_src[i]))
    betaTwo.append(r[i]*np.sin(theta_src[i]))
'''


# randomized shear vals b/w 0 and 0.1 and ellipticity vals between 0 and 0.5

# Define the distributions, following Oguri & Marshall 2010MNRAS.405.2579O
# (https://ui.adsabs.harvard.edu/abs/2010MNRAS.405.2579O/abstract)
# Shear follows a lognormal distribution with mean 0.05 and dispersion 0.2 dex
# Ellipticity follows a truncated Gaussian distribution with mean 0.3 and standard deviation 0.16, but truncated at $e=0$ and $e=0.9$

def random_shear(nran,mu=0.05,sg=0.2):
    return np.random.lognormal(mean=np.log10(mu)/np.log10(np.e),sigma=sg/np.log10(np.e),size=nran)

def random_ellip(nran,mu=0.3,sg=0.16,ehi=0.9):
    # note: in order to ensure that we have enough values after truncation, we initially draw extra values
    tmp = np.random.normal(loc=mu,scale=sg,size=2*nran)
    # do the truncation
    tmp = tmp[(tmp>=0)&(tmp<0.9)]
    # return the first nran values from the truncated array
    return tmp[:nran]

shear_vals = random_shear(num_mock)
ellip_vals = random_ellip(num_mock)

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