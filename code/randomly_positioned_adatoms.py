#!/usr/bin/env python
# coding: utf-8

# # Graphene with N randomly positioned adatoms

# In[1]:


import kwant
import tinyarray
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


# ## Definition of the Graphene strip

# Let's start by defining a system without SOC terms, but with two orbitals per site representing the spin. To do so, we've adopted the `general` lattice definition from `kwant.lattice` module. Such a function needs the primitive vectors and the coordinates of the basis atoms:
# 
# **Primitive vectors:**
# * (1, 0)
# * ($\sin 30°$, $\cos 30°$)
# 
# **Coordinates of basis atoms:**
# * (0, 0)
# * (0, $\frac{1}{\sqrt 3}$)

# In[2]:


zeros_2x2 = tinyarray.array([[0,0],[0,0]])
sigma_0 = tinyarray.array([[1,0],[0,1]])
sigma_x = tinyarray.array([[0,1],[1,0]])
sigma_y = tinyarray.array([[0,-1j],[1j,0]])
sigma_z = tinyarray.array([[1,0],[0,-1]])


# ### Builder function:

# In[3]:


class Rectangle:
    """
    Class to define callable objects to define the 
    shape of the scattering region of a rectangular
    system.
    """
    def __init__(self, W, L):
        '''
        Calling the scattering region as strip:
        W = width of the strip
        L = length of the strip 
        '''
        self.W = W
        self.L = L
        
    def __call__(self, pos):
        W, L = self.W, self.L
        x, y = pos
        return -W/2 < y < W/2 and -L/2 <= x <= L/2
    
    def leads(self, pos):
        W = self.W
        _, y = pos
        return -W/2 < y < W/2

def make_graphene_strip(lattice, scatter_shape, hopping=1, on_site=0):    
    syst = kwant.Builder()
    syst[lattice.shape(scatter_shape, (0, 0))] = on_site

    # specify the hoppings of the graphene lattice in the
    # format expected by builder.HoppingKind
    a, b = lattice.sublattices
    hoppings = (((0, 0), a, b), ((0, 1), a, b), ((-1, 1), a, b))
    syst[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = hopping
    include_ISOC(syst, [a,b], lambda_I=1)
    return syst


def make_graphene_leads(lattice, lead_shape, hopping=1, on_site=0):
    a, b = lattice.sublattices
    symmetry = kwant.TranslationalSymmetry((-1,0))
    symmetry.add_site_family(a, other_vectors=[(-1,2)])
    symmetry.add_site_family(b, other_vectors=[(-1,2)])
    
    lead_0 = kwant.Builder(symmetry)
    lead_0[lattice.shape(lead_shape, (0,0))] = on_site
    
    hoppings = (((0, 0), a, b), ((0, 1), a, b), ((-1, 1), a, b))
    lead_0[(kwant.builder.HoppingKind(*hopping) for hopping in hoppings)] = hopping
    include_ISOC(lead_0, [a,b], lambda_I=1)
    
    lead_1 = lead_0.reversed()
    return [lead_0, lead_1]


def include_ISOC(system, G_sub_lattices, lambda_I=1):
    """
    ## INCLUDING THE INTRINSIC SOC (isoc):
    system         := kwant.builder.Builder
    G_sub_lattices := list of Graphene sublattices
    lambda_I       := parameter  
    """
    sub_a, sub_b = G_sub_lattices
    # lambda_I   = 1 ## non-realistic; see reference: PRL 110, 246602 (2013)
    H_isoc = -1j/3 * lambda_I/np.sqrt(3) * sigma_z # counter-clockwise
    system[kwant.builder.HoppingKind((1,0), sub_a, sub_a)]  =  H_isoc
    system[kwant.builder.HoppingKind((0,1), sub_a, sub_a)]  = -H_isoc
    system[kwant.builder.HoppingKind((-1,1), sub_a, sub_a)] =  H_isoc
    system[kwant.builder.HoppingKind((1,0), sub_b, sub_b)]  =  H_isoc
    system[kwant.builder.HoppingKind((0,1), sub_b, sub_b)]  =  H_isoc
    system[kwant.builder.HoppingKind((-1,1), sub_b, sub_b)] = -H_isoc


# In[4]:


sin_30 = 1/2
cos_30 = np.sqrt(3)/2

graphene = kwant.lattice.general([(1, 0), (sin_30, cos_30)], # primitive vectors
                                 [(0, 0), (0, 1 / np.sqrt(3))],        # coord. of basis atoms
                                 norbs = 2,                            # number of orbitals per site (spin)
                                 name='Graphene'                       # name of identification
                                )
## Split in sublattices
A, B = graphene.sublattices

shape = Rectangle(W=2.5, L=5)
system = make_graphene_strip(graphene, shape, on_site=0*sigma_0, hopping=sigma_0)
leads  = make_graphene_leads(graphene, shape.leads, on_site=0*sigma_0, hopping=sigma_0)

for lead in leads:
    system.attach_lead(lead)


# In[5]:


def family_colors(site):
    return 'w' if site.family == A else 'k' if site.family == B else 'r'

def hopping_colors(site1, site2):
#         if (site1.family==A and site1.family==site2.family) and (site1.tag == np.array([0,0]) or site2.tag == np.array([0,0])): # and site1.tag == np.array([0,0]) and site1.family==A:
#             color = 'red'
        if site1.family == site2.family:
            color='blue'
        else:
            color='black'
        return color

def hopping_lw(site1, site2):
    return 0.04 if site1.family == site2.family else 0.1


# In[6]:


## Figure
fig, ax = plt.subplots(figsize=(20,5))
kwant.plot(system,
           site_color=family_colors,
           hop_color=hopping_colors,
           hop_lw=hopping_lw,
           site_lw=0.1, ax=ax)
ax.set_aspect('equal')
plt.show()


# ## Randomly placing the adatoms

# We want to place, in a random fashion, N Hydrogen-atoms on top of Graphene sites. As a criteria, we want to avoid the sites at the interface between the scattering system and the leads. It can be interesting by now to avoid the boundaries of the strip all soever.
# 
# Since our system present altered hoppings between next-nearest-neighbors, the sites allowed to host H-adatoms has to be at least one primitive vector away from the boundaries of the system. Now our task is
# * to collect the sites of the system, 
# * exclude those that are too close of the boundaries, and then...
# * to sample from the remaining, N locations for the adatoms. 

# By inspection, one can see that the limits are different for each of the sublattices. 
# For the sublattice **A** we have the following *tags*: 

# In[7]:


first_component_tags_sublattice_A = np.unique([site.tag[0] for site in system.sites() if site.family==A])
second_component_tags_sublattice_A = np.unique([site.tag[1] for site in system.sites() if site.family==A])

print("First component of the tags for sublattice A sites : \n",
      first_component_tags_sublattice_A)
print("Second component of the tags for sublattice A sites : \n",
      second_component_tags_sublattice_A)


# In[8]:


A(-12,0) in system


# In[9]:


# dir(system)


# While the tags for the sublattice **B** are composed by the values: 

# In[10]:


first_component_tags_sublattice_B = np.unique([site.tag[0] for site in system.sites() if site.family==B])
second_component_tags_sublattice_B = np.unique([site.tag[1] for site in system.sites() if site.family==B])
print("First component of the tag for sublattice B sites : \n",
      first_component_tags_sublattice_B)
print("Second component of the tag for sublattice B sites : \n",
      second_component_tags_sublattice_B)


# Excluding the sites at the boundaries and making the pairs of values:

# In[11]:


A_allowed_sites_tags = [(i,j) for i,j in 
                        product(first_component_tags_sublattice_A[2:-2], 
                                second_component_tags_sublattice_A[1:-1])
                       ]


# In[12]:


B_allowed_sites_tags = [(i,j) for i,j in 
                        product(first_component_tags_sublattice_B[2:-2], 
                                second_component_tags_sublattice_B[1:-1])
                       ]
# print(B_allowed_sites_tags)


# Choosing N_A sites out of sublattice-A, and N_B sites out of sublattice-B.

# In[13]:


## To get just one site out of 'A':
N_A_tags = len(A_allowed_sites_tags) # number of allowed sites
print(N_A_tags)
A_allowed_sites_tags[np.random.choice(N_A_tags)] # the chosen site 


# In[14]:


## To get just one site out of 'B': 
N_B_tags = len(B_allowed_sites_tags) # number of allowed sites
B_allowed_sites_tags[np.random.choice(N_B_tags)] # the chosen site


# In[15]:


np.random.seed(0) ## allow reproducibility
N_A_sample = 2
A_sites_index_list = np.random.choice(N_A_tags, size=N_A_sample, replace=False)
A_sample = [A_allowed_sites_tags[i] for i in A_sites_index_list]
print(A_sample)


# In[16]:


np.random.seed(1) ## allow reproducibility
N_B_sample = 2
B_sites_index_list = np.random.choice(N_B_tags, size=N_B_sample, replace=False)
B_sample = [B_allowed_sites_tags[i] for i in B_sites_index_list]
print(B_sample)


# ### Defining the "pseudo-lattice" for H-atoms:

# In[17]:


hydrogen = kwant.lattice.general([(1, 0), (sin_30, cos_30)], # primitive vectors
                                 [(0, 0), (0, 1 / np.sqrt(3))],        # coord. of basis atoms
                                 norbs = 2,                            # number of orbitals per site (spin)
                                 name='H'                              # name for identification
                                )
## Split in sublattices
HA, HB = hydrogen.sublattices


# In[18]:


T = 1

for tagA in A_sample:
    system[HA(*tagA)] = zeros_2x2   # on-site
    system[A(*tagA), HA(*tagA)] = T # hopping with C_H

for tagB in B_sample:
    system[HB(*tagB)] = zeros_2x2   # on-site
    system[B(*tagB), HB(*tagB)] = T # hopping with C_H


# In[19]:


def family_colors_H(site):
        if site.family == A:
            color = 'k'
        elif site.family == B:
            color = 'k'
        elif site.family == HA:
            color = 'cyan'
        elif site.family == HB:
            color = 'cyan'
        else:
            color = 'cyan'
        return color


# In[20]:


## Figure
fig, ax = plt.subplots(figsize=(20,5))
kwant.plot(system,
           site_color=family_colors_H,
           hop_color=hopping_colors,
           hop_lw=hopping_lw,
           site_lw=0.1, ax=ax)
ax.set_aspect('equal')
plt.show()


# $$
# H_{ASO} = \frac{i}{3}\sum_{\langle\langle C_H, j \rangle\rangle} A^{\dagger}_{\sigma}c_{j,\sigma^{\prime}} \left[ \frac{\Lambda_I}{\sqrt{3}} \nu_{C_H, j} \hat{s}_z \right]_{\sigma, \sigma^{\prime}}
# $$
# 
# 
# $$
# H_{BR} = \frac{2i}{3} \sum_{\langle C_H, j \rangle} A^{\dagger}_{\sigma}B_{j\sigma^{\prime}} \left[\Lambda_{BR} (\hat{s} \times \vec{d}_{C_H, j})_z\right]_{\sigma,\sigma^{\prime}}
# $$
# 
# $$
# H_{PIA} = \frac{2i}{3} \sum_{\langle\langle i,j \rangle\rangle} B^{\dagger}_{i\sigma}B_{j\sigma^{\prime}}[\Lambda_{PIA}(\hat{s}\times\vec{D}_{ij})_{z}]_{\sigma\sigma^{\prime}}
# $$
# 
# [PRL 110, 246602 (2013)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.246602) 

# ## Getting the neighbors of the C_H's

# To properly define the hoppings, we have to know where the adatoms are located and to keep track of its neighbors, the nearest (NN) and the next-nearest (NNN). It is important to pay attention to the situation in which the adatoms are placed on top of neighboring Carbon atoms. If the code are implemented in such a way that the different SOC contributions are added one-by-one, the hoppings between the neighbors $C_H$s might be updated wrongly, counting the same SOC contribution twice (or even more).
# 
# Let's do a brainstorm listing the steps required to define all hopping terms of the system:
# 
# 1. Define the Graphene strip and the leads:
#     - Carbon (C) atoms on-site energy
#     - "A-B" hoppings (NN)
#     - "A-A" and "B-B" hoppings (NNN) **ISO**
# 2. Place the H-adatoms:
#     - Hydrogen atoms on-site ennergy
#     - "H-C" hoppings
# 3. Get the neighbors of all the Carbon atom coupled to a H-atom ($C_H$):
#     - Nearest-Neighbors (NN): different sublattices (for **BR** and **PIA**)
#     - Next-Nearest-Neighbors (NNN): same sublattices (for **ASO**)
# 4. With the positions of $C_H$ and the their neighbors change the hoppings: **paying attention to duplicates**
#     - List of $C_H$-sites
#     - Two lists of neighbors for each of the $C_H$: NN's and NNN's.
#     - **For each soc contribution** define a `list` or a `set` (updated pairs) on which we're going to put every pair of sites we have updated: The **PIA** pair of one site may form a NNN pair (**ASO**)  
#     - Before changing the hopping verify if the pair of sites is already on the list.
#     - Change the hoppings if the pair isn't in the list,...
#     - ...then add the pair (site1,site2) and (site2,site1) on the list.  

# In[21]:


# for s in system.sites():
#     if s.family==HA or s.family==HB:
#         print(s.family.name)
#         print('tag = ', s.tag)
#         print('type(tag) = ', type(s.tag))
#         print(type(s.pos), end='\n\n')


# In[22]:


def get_CH(system, CH_sublattices, H_sublattices):
    """
    system := kwant.builder.Builder
    CH_sublattices := list of the graphene sublattices
    H_sublattices := list of H atoms sublattices
    """
    A, B = CH_sublattices
    HA, HB = H_sublattices
    
    list_CH = []
    
    for site in system.sites():
        if site.family == HA:
            list_CH.append(A(*site.tag))
        elif site.family == HB:
            list_CH.append(B(*site.tag))
    return list_CH

def get_neighbors(system, C_H_site, CH_sublattices):
    site_tag = C_H_site.tag
    site_sub = C_H_site.family
    nn_list = get_nn(system, site_tag, site_sub, CH_sublattices)
    nnn_list = get_nnn(system, site_tag, site_sub)
    return [nn_list, nnn_list]

def get_nn(system, tag, sub_s, list_sub_lat):
    """
    system := kwant.builder.Builder
    tag    := tuple (i,j) of the site's tag
    sub_s  := site's sublattice
    
    Notice that 
    """
    list_sub_lat.remove(sub_s)
    sub_nn, = list_sub_lat
    # print(sub_nn.name[-1])
    name_indx = int(sub_s.name[-1])
    delta = +1 if name_indx == 0 else -1
#     print(delta)
    i,j = tag
    nn_tag_list = [(i,j), (i+delta,j-delta), (i,j-delta)] 
    nn_sites = [
        sub_nn(*t) for t in nn_tag_list if sub_nn(*t) in system
    ]
#     for site in nn_sites:
#         print(site)
    return nn_sites
        
def get_nnn(system, tag, sub_s):
    """
    system := kwant.builder.Builder
    tag    := tuple (i,j) of the site's tag
    sub_s  := site's sublattice
    
    Notice that 
    """
    #sub_nnn = sub_s

    i,j = tag
    nnn_tag_list = [(  i,j+1), (i+1,  j),
                    (i+1,j-1), (  i,j-1),
                    (i-1,  j), (i-1,j+1)] 
    nnn_sites = [
        sub_s(*t) for t in nnn_tag_list if sub_s(*t) in system
    ]
#     for site in nnn_sites:
#         print(site)
    return nnn_sites


# In[23]:


CH_sites = get_CH(system, [A,B], [HA,HB])

for site in CH_sites:
    print(site)


# In[24]:


all_neighbors = [get_neighbors(system, CH, [A,B]) for CH in CH_sites]

all_NN_neighbors = [a[0] for a in all_neighbors]
all_NNN_neighbors = [a[1] for a in all_neighbors]


# ## Including the SOC terms induced by the adatoms

# In[25]:


def include_ASO_sitewise(system, CH_site, NNN_site, hop_list, Lambda_I=1):
    """
    This function has two effects:
        1. Define a hopping between CH_site and NNN_site
        2. Returns an updated hop_list
    
    Naturaly, the first effect occurs only in the case the hopping 
    isn't on the list. Otherwise the hopping will not be included and
    the hop_list will not be altered.
    """
    # 1. Verify if the hopping is duplicated
    if (CH_site, NNN_site) not in hop_list: 

        # 2.1. Identify the hopping 1/2
        delta_tag = list(NNN_site.tag - CH_site.tag)
        if delta_tag in ([0,1], [1,-1], [-1,0]): sign = -1
        else: sign = +1
        
        # 2.2. Identify the hopping 2/2
        family_id  = 1 - 2 * int(CH_site.family.name[-1]) ## 1 (-1) if sublattice == A (B)
        
        # 3. Define the hopping
        H_asoc = sign * family_id * 1j/3 * Lambda_I/np.sqrt(3) * sigma_z # clockwise    
        system[CH_site, NNN_site] = H_asoc ## sytem[sublat_1(target), sublat_2(source)]
        
        # 4. Save pairs in hop_list
        hop_list.append((CH_site, NNN_site)) # (CH, NNN)
        hop_list.append((NNN_site, CH_site)) # (NNN, CH)
        
    return hop_list


def include_BR_sitewise(system, CH_site, NN_site, hop_list, Lambda_BR=1):
    """
    This function has two effects:
        1. Define a hopping between CH_site and NN_site
        2. Returns an updated hop_list
    
    The first effect occurs only in the case the hopping 
    isn't on the list. Otherwise the hopping will not be included and
    the hop_list will not be altered.
    """
    # 1. Verify if the hopping is duplicated
    if (CH_site, NN_site) not in hop_list:
        
        # 2. Identify the hopping 2/2:
        dx, dy = np.sqrt(3) * (CH_site.pos - NN_site.pos) 
        H_BR = (2j/3) * Lambda_BR * (dy * sigma_x - dx * sigma_y) ## (S X d_ij)_z 
        
        # 3. Define the hopping
        system[CH_site, NN_site] += H_BR
        
        # 4. Save pairs in hop_list
        hop_list.append((CH_site, NN_site)) # (CH, NNN)
        hop_list.append((NN_site, CH_site)) # (NNN, CH)
        
    return hop_list


def include_PIA_sitewise(system, site_target, site_source, hop_list, Lambda_PIA=1):
    """
    This function has two effects:
        1. Define a hopping between CH_site and NN_site
        2. Returns an updated hop_list
    
    The first effect occurs only in the case the hopping 
    isn't on the list. Otherwise the hopping will not be included and
    the hop_list will not be altered.
    """
    # 1. Verify if the hopping is duplicated
    if (site_target, site_source) not in hop_list:
        
        # 2. Identify the hopping:
        Dx, Dy = site_target.pos - site_source.pos
        
        # 3. Define the hopping
        H_PIA =  2j/3 * Lambda_PIA * (Dy * sigma_x - Dx * sigma_y) ## (S x D_ij)_z
        system[site_target, site_source] += H_PIA
        
        # 4. Save pairs in hop_list
        hop_list.append((site_target, site_source)) # (site1, site2)
        hop_list.append((site_source, site_target)) # (site2, site1)
        
    return hop_list


# #### Adatom induced spin-orbit coupling (ASO)

# In[26]:


## Calculate and include the Adatom induced spin-orbit coupling
hop_list_ASO = []
for CH_site, NNN_sites in zip(CH_sites, all_NNN_neighbors):
    for NNN_site in NNN_sites:
#         print(CH_site,'--', NNN_site)
        include_ASO_sitewise(system, CH_site, NNN_site, hop_list_ASO, Lambda_I=1)


# In[27]:


## Figure
fig, ax = plt.subplots(figsize=(20,5))
kwant.plot(system,
           site_color=family_colors_H,
           hop_color=hopping_colors,
           hop_lw=hopping_lw,
           site_lw=0.1, ax=ax)
ax.set_aspect('equal')
plt.show()


# #### Bychkov-Rashba spin-orbit coupling (BR)

# In[28]:


## Calculate and include into the system the Bychkov-Rashba spin-orbit coupling (BR)
hop_list_BR = []
for CH_site, NN_sites in zip(CH_sites, all_NN_neighbors):
    print(len(NN_sites))
    for NN_site in NN_sites:
        print(NN_site)
        include_BR_sitewise(system, CH_site, NN_site, hop_list_BR, Lambda_BR=1)


# In[29]:


## Figure
fig, ax = plt.subplots(figsize=(20,5))
kwant.plot(system,
           site_color=family_colors_H,
           hop_color=hopping_colors,
           hop_lw=hopping_lw,
           site_lw=0.1, ax=ax)
ax.set_aspect('equal')
plt.show()


# ### Pseudo-spin inversion asymmetry SOC (PIA)

# In[30]:


## Calculate and include into the system the Pseudo-spin inversion asymmetry spin-orbit coupling (PIA)
hop_list_PIA = []
for NN_sites in all_NN_neighbors:
    targets = [NN_sites[(i+1)%3] for i in range(3)]
    for site1, site2 in zip(targets, NN_sites):
        print(site1, '<--', site2)
        include_PIA_sitewise(system, site1, site2, hop_list_PIA, Lambda_PIA=1)


# In[31]:


## Figure
fig, ax = plt.subplots(figsize=(20,5))
kwant.plot(system,
           site_color=family_colors_H,
           hop_color=hopping_colors,
           hop_lw=hopping_lw,
           site_lw=0.1, ax=ax)
ax.set_aspect('equal')
plt.show()


# ## Checking the hoppings

# Since we have fixed the seed for both of the sampling processes (one for HA positions and other for HB positions), we can treat the specific hoppings and check if their values are correct. Let's focus on the hoppings surrounding or point towards the site at the position $x=0$ and $y=0$.
# 
# $$
# [A_{00}, B_{0-1}] = t \begin{pmatrix} 1 & 0\\0 & 1 \end{pmatrix} + \Lambda_{BR} \left[ \frac{1}{3}\begin{pmatrix} 0 & i\\i & 0 \end{pmatrix} - \sqrt{3}\begin{pmatrix} 0 & 1\\-1 & 0 \end{pmatrix} \right]
# $$

# In[37]:


print(system[A(0,0), B(0,-1)]) ## Has to be identical to the matrix above


# In[38]:


print(system[B(0,-1), A(0,0)]) ## Has to be the adjunt of the previous matrix


# ## Considerations:

# The model wasn't thought to be used in a system with high concentration of H-atoms, the exercise here was good to see the difficulties to define such a system. The concentration aimed is something lower than 10% with very sparsed location for the H-atom: 1 H-atom for 10 C-atoms without overlap. From here, we have to address the following scenarios:
# 
# * Graphene with ISO
# * Graphene + ISO + Adatom at the Center
# * Graphene + ISO + Adatom at the Center + Magnetic Field
# * Graphene + ISO + Adatom at the Center + Magnetic Field + Magnetic moment

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




