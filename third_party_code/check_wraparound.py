import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.constants

import kwant
import sys
import kwant.wraparound as wraparound

# -------------------------------------------------------------------
# Constants
qe = sp.constants.value("elementary charge") # unit: C
me = sp.constants.value("electron mass")/qe*1e-18 #unit: eV*s^2/nm^2
hP = sp.constants.value("Planck constant in eV s") #unit: eV*s
hbar = hP/(2*sp.pi) #unit: eV*s
# -------------------------------------------------------------------

a = 0.03
V0 = 0.0

W = 60
L = 50

t = hbar**2/(2*me*a**2) # units: eV


#---------------------------------------------first without PBC (no wraparound)-----------------------------------------------------------------------#
# lat = kwant.lattice.square(a)
#
# # Infinite potential plane in y direction
# syst = kwant.Builder()#kwant.TranslationalSymmetry(lat.vec((0, W))))
# syst[(lat(i,j) for i in range(L) for j in range(W))] = lambda p: 4*t
# syst[lat.neighbors(1)] = -t
#
# #syst = wraparound.wraparound(syst)
#
# lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))#, lat.vec((0, W))))
# lead[(lat(0,j) for j in range(W))] = 4*t
# lead[lat.neighbors(1)] = -t
#
# #lead = wraparound.wraparound(lead, keep=0)
#
# syst.attach_lead(lead)
# syst.attach_lead(lead.reversed())
#
# kwant.plot(syst)
#
# syst = syst.finalized()
#
# # -------------------------------------------------------
# # Calculation
#
# energies = np.arange(0.0, 5.0, 0.05)
# transmission = []
# num_prop = []
# for energy in energies:
#     smatrix = kwant.smatrix(syst, energy)
#     transmission.append(smatrix.transmission(1, 0))
#     num_prop.append(smatrix.num_propagating(0))
# # -------------------------------------------------------
#
# # Plot transmission and propagating modes
# plt.plot(energies, transmission, '.')
# plt.show()
#
# plt.plot(energies, num_prop, '.')
# plt.show()
#
# # Plot wave function squared for the first mode for a specified energy and ky
# wf = kwant.solvers.default.wave_function(syst, energy=2.0, params=[0.0])
# kwant.plotter.map(syst, np.real(wf(0)[0]), fig_size=(8,5))
#


####################################################infinite plane##############################################################################
lat = kwant.lattice.square(a)

# Infinite potential plane in y direction
syst = kwant.Builder(kwant.TranslationalSymmetry(lat.vec((0, W))))
syst[(lat(i,j) for i in range(L) for j in range(W))] = lambda p: 4*t
syst[lat.neighbors(1)] = -t

syst = wraparound.wraparound(syst)

lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0), lat.vec((0, W))))
lead[(lat(0,j) for j in range(W))] = 4*t
lead[lat.neighbors(1)] = -t

lead = wraparound.wraparound(lead, keep=0)

syst.attach_lead(lead)
syst.attach_lead(lead.reversed())

kwant.plot(syst)

syst = syst.finalized()

# -------------------------------------------------------
# Calculation
ky = 0 #-np.pi/2

# energies = np.arange(0.0, 5.0, 0.05)
# transmission = []
# num_prop = []
# for energy in energies:
#     smatrix = kwant.smatrix(syst, energy, [ky])
#     # smatrix = kwant.smatrix(syst, energy, params=dict(k_x=None, k_y=ky))
#     transmission.append(smatrix.transmission(1, 0))
#     num_prop.append(smatrix.num_propagating(0))
# # -------------------------------------------------------
#
# # Plot transmission and propagating modes
# fig, ax = plt.subplots(ncols=2, figsize=(12,6))
# ax[0].plot(energies, transmission, '.')
# ax[1].plot(energies, num_prop, '.')
# plt.show()

# Plot wave function squared for the first mode for a specified energy and ky
wf = kwant.solvers.default.wave_function(syst, energy=2.0, args=[ky])
# wf = kwant.solvers.default.wave_function(syst, energy=2.0, params=dict(k_x=None, k_y=ky))
psi = wf(0)[0]
max_val_dens = np.max(np.abs(psi)**2)
min_val_dens = np.min(np.abs(psi)**2)


min_real = np.min(psi.real)
max_real = np.max(psi.real)

print('\t MINIMUM VALUE OF PSI.REAL = ', min_real)
print('\t MAXIMUM VALUE OF PSI.REAL = ', max_real)

fig, ax = plt.subplots(ncols=3, figsize=(12,6))
kwant.plotter.map(syst, np.real(psi), ax=ax[0], show=False)
kwant.plotter.map(syst, np.imag(psi), ax=ax[1], show=False)
kwant.plotter.map(syst, np.abs(psi)**2, ax=ax[2], show=False, vmax=max_val_dens, vmin=0)
plt.show()
