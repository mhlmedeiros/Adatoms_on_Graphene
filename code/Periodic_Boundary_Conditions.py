#!/usr/bin/env python
# coding: utf-8

# # How to build a Graphene system with PBC?

# In[2]:


import kwant
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.sparse.linalg as sla
import numpy as np


# In[3]:


def make_system(e=0, t=1):
    lat = kwant.lattice.honeycomb()
    sym = kwant.TranslationalSymmetry(lat.vec((10, 0)), lat.vec((0, 10)))
#     sym = kwant.TranslationalSymmetry(lat.vec((0, 1)))

    # Build ancillary system with 2d periodic BCs.  This system cannot be
    # finalized in Kwant 1.0.
    anc = kwant.Builder(sym)
    anc[lat.shape(lambda p: True, (0, 0))] = None
    anc[lat.neighbors()] = None

    # Build a site-by-site and hopping-by-hopping copy of the ancillary system,
    # but this time without automatic PBCs.  This system can be finalized.
    sys = kwant.Builder()
    sys[anc.sites()] = e
    sys[((a, sym.to_fd(b)) for a, b in anc.hoppings())] = -t
    return sys

sys = make_system()
kwant.plot(sys);


# In[35]:


fsystem = sys.finalized()
spectrum = kwant.kpm.SpectralDensity(fsystem)
spectrum.add_vectors(500)


# In[36]:


energies_subset = np.linspace(-1.5,1.5,201)
densities_subset = spectrum(energies_subset)
energies, densities = spectrum()


# In[37]:


plt.plot(energies, densities.real)
plt.plot(energies_subset, densities_subset.real)


# In[7]:


ham_mat = fsystem.hamiltonian_submatrix(sparse=True)

# we only calculate the 15 lowest eigenvalues
ev = sla.eigsh(ham_mat.tocsc(), k=15, sigma=0, return_eigenvectors=False)


# In[8]:


ev.shape


# In[33]:


plt.plot(ev)


# In[21]:


lat = kwant.lattice.honeycomb()
sym = kwant.TranslationalSymmetry(lat.vec((10, 0)), lat.vec((0, 10)))


anc = kwant.Builder(sym)
anc[lat.shape(lambda p: True, (0, 0))] = 0
anc[lat.neighbors()] = -1


# In[22]:


syst_wrapped = kwant.wraparound.wraparound(anc, keep=1).finalized()


# In[23]:


kx_list = np.linspace(-np.pi, np.pi, 201)


def h_k(kx):
    p = dict(k_x=kx)
    return syst_wrapped.hamiltonian_submatrix(params=p)


energy_tb = np.array([la.eigvalsh(h_k(k)) for k in kx_list])


# In[29]:


plt.figure(figsize=(15,5))
plt.plot(kx_list, energy_tb)
# plt.xlim(-2,2)
plt.ylim(-0.7,0.7)
plt.show()


# In[ ]:




