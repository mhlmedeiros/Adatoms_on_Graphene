import kwant
import tinyarray
import numpy as np
import matplotlib.pyplot as plt


## Matrix Definitions:
zeros_2x2 = tinyarray.array([[0,0],[0,0]])
sigma_0 = tinyarray.array([[1,0],[0,1]])
sigma_x = tinyarray.array([[0,1],[1,0]])
sigma_y = tinyarray.array([[0,-1j],[1j,0]])
sigma_z = tinyarray.array([[1,0],[0,-1]])

## Graphene lattice Definition:
sin_30 = 1/2
cos_30 = np.sqrt(3)/2

graphene = kwant.lattice.general([(1, 0), (sin_30, cos_30)],    # primitive vectors
                                 [(0, 0), (0, 1 / np.sqrt(3))], # coord. of basis atoms
                                 norbs = 2,                     # number of orbitals per site (spin)
                                 name='Graphene')               # name of identification
## Split in sublattices
A, B = graphene.sublattices

## Adatom lattice definition:
hydrogen = kwant.lattice.general([(1, 0), (sin_30, cos_30)],    # primitive vectors
                                 [(0, 0), (0, 1 / np.sqrt(3))], # coord. of basis atoms
                                 norbs = 2,                     # number of orbitals per site (spin)
                                 name='H')                      # name for identification

## Split in sublattices
HA, HB = hydrogen.sublattices


## Shape:
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

## Builders without magnetic field:
def make_graphene_strip(lattice, scatter_shape, t=1, on_site=0, iso=1e-6):
    syst = kwant.Builder()
    syst[lattice.shape(scatter_shape, (0, 0))] = on_site

    # specify the hoppings of the graphene lattice in the
    # format expected by builder.HoppingKind
    a, b = lattice.sublattices
    hoppings_list = (((0, 0), a, b), ((0, 1), a, b), ((-1, 1), a, b))
    hopping = t * sigma_0
    syst[[kwant.builder.HoppingKind(*hop) for hop in hoppings_list]] = hopping
    syst.eradicate_dangling()
    include_ISOC(syst, [a,b], lambda_I=iso)
    return syst

def make_graphene_leads(lattice, lead_shape, t=1, on_site=0, iso=1e-6):
    a, b = lattice.sublattices
    symmetry = kwant.TranslationalSymmetry((-1,0))
    symmetry.add_site_family(a, other_vectors=[(-1,2)])
    symmetry.add_site_family(b, other_vectors=[(-1,2)])

    lead_0 = kwant.Builder(symmetry)
    lead_0[lattice.shape(lead_shape, (0,0))] = on_site

    hoppings_list = (((0, 0), a, b), ((0, 1), a, b), ((-1, 1), a, b))
    hopping = t * sigma_0
    lead_0[(kwant.builder.HoppingKind(*hop) for hop in hoppings_list)] = hopping
    lead_0.eradicate_dangling()
    include_ISOC(lead_0, [a,b], lambda_I=iso)

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


## Inserting adatom:
def insert_adatom(syst, pos_tag, sub_lat,  T=1, eps_h=1, L_I=1, L_BR=1, L_PIA=1):
    if sub_lat == A:
        site_CH = A(*pos_tag)
        site_H = HA(*pos_tag)
    elif sub_lat == B:
        site_CH = B(*pos_tag)
        site_H = HB(*pos_tag)

    ## On-site:
    syst[site_H] = eps_h * sigma_0
    ## Hopping:
    syst[site_H, site_CH] = T * sigma_0
    ## Neighbors
    nn_sites, nnn_sites = get_neighbors(syst, site_CH)

    ## Calculate and include the Adatom induced spin-orbit coupling
    for site in nnn_sites:
        include_ASO_sitewise(syst, site_CH, site, Lambda_I=L_I)

    ## Calculate and include into the system the Bychkov-Rashba spin-orbit coupling (BR)
    for site in nn_sites:
        include_BR_sitewise(syst, site_CH, site, Lambda_BR=L_BR)

    ## Calculate and include into the system the Pseudo-spin inversion asymmetry spin-orbit coupling (PIA)
    targets = [nn_sites[(i+1)%3] for i in range(3)]
    for site1, site2 in zip(targets, nn_sites):
        include_PIA_sitewise(syst, site1, site2, Lambda_PIA=L_PIA)


## Getting the neighboring sites
def get_neighbors(system, C_H_site):
    site_tag = C_H_site.tag
    site_sub = C_H_site.family
    nn_list = get_nn(system, site_tag, site_sub)
    nnn_list = get_nnn(system, site_tag, site_sub)
    return nn_list, nnn_list

def get_nn(system, tag, sub_s):
    """
    system := kwant.builder.Builder
    tag    := tuple (i,j) of the site's tag
    sub_s  := site's sublattice

    Notice that
    """
    list_sub_lat = [A, B]
    list_sub_lat.remove(sub_s)
    sub_nn, = list_sub_lat
    # print(sub_nn.name[-1])
    name_indx = int(sub_s.name[-1])
    delta = +1 if name_indx == 0 else -1
    i,j = tag
    nn_tag_list = [(i,j), (i+delta,j-delta), (i,j-delta)]
    nn_sites = [
        sub_nn(*t) for t in nn_tag_list if sub_nn(*t) in system
    ]
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


## Changing the hoppings around the adatom
def include_ASO_sitewise(system, CH_site, NNN_site, Lambda_I=1):
    """
    Define and replace the hopping between CH_site and one of its NNN sites,
    which is identified here as NNN_site.
    """
    # 1.1. Identify the hopping 1/2
    delta_tag = list(NNN_site.tag - CH_site.tag)
    if delta_tag in ([0,1], [1,-1], [-1,0]): sign = -1
    else: sign = +1

    # 1.2. Identify the hopping 2/2
    family_id  = 1 - 2 * int(CH_site.family.name[-1]) ## 1 (-1) if sublattice == A (B)

    # 3. Define the hopping
    H_asoc = sign * family_id * 1j/3 * Lambda_I/np.sqrt(3) * sigma_z # clockwise
    system[CH_site, NNN_site] = H_asoc

def include_BR_sitewise(system, CH_site, NN_site, Lambda_BR=1):
    """
    Update the hopping between the CH_site and the NN_site to include
    the Bychkov-Rashba SOC.
    """
    # 1. Identify the hopping 2/2:
    dx, dy = np.sqrt(3) * (CH_site.pos - NN_site.pos)
    H_BR = (2j/3) * Lambda_BR * (dy * sigma_x - dx * sigma_y) ## (S X d_ij)_z

    # 2. Define the hopping
    system[CH_site, NN_site] += H_BR

def include_PIA_sitewise(system, site_target, site_source, Lambda_PIA=1):
    """
    Define the PIA hopping and add to the already existent hopping between
    site_target and site_source.
    """
    # 1. Identify the hopping:
    Dx, Dy = site_target.pos - site_source.pos

    # 2. Define the hopping
    H_PIA =  2j/3 * Lambda_PIA * (Dy * sigma_x - Dx * sigma_y) ## (S x D_ij)_z
    system[site_target, site_source] += H_PIA


## Plotting functions:
def family_colors(site):
    return 'w' if site.family == A else 'k' if site.family == B else 'r'

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
            color = 'r'
        return color

def hopping_colors(site1, site2):
#         if (site1.family==A and site1.family==site2.family) and (site1.tag == np.array([0,0]) or site2.tag == np.array([0,0])): # and site1.tag == np.array([0,0]) and site1.family==A:
#             color = 'red'
        if site1.family == site2.family:
            color='grey'
        else:
            color='black'
        return color

def hopping_lw(site1, site2):
    return 0.04 if site1.family == site2.family else 0.1

def plot_system(system, ax=None):
    if ax== None: fig, ax = plt.subplots(figsize=(20,5))
    kwant.plot(system,
           site_color=family_colors_H,
           hop_color=hopping_colors,
           hop_lw=hopping_lw,
           site_lw=0.1, ax=ax)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=20)
    plt.show()

def plot_bands(lead, ax=None):
    lead = lead.finalized()
    if ax == None: fig, ax = plt.subplots(figsize=(7,5))
    kwant.plotter.bands(lead, show=False, ax=ax, params={'colors':'k'})
    ax.set_xlabel("momentum [(lattice constant)^-1]")
    ax.set_ylabel("energy [t]")
    ax.tick_params(labelsize=20)
    plt.show()

## Conductance
def calculate_conductance(syst, energy_values):
    syst = syst.finalized()
    data = []
    for energy in energy_values:
        # compute the scattering matrix at a given energy
        smatrix = kwant.smatrix(syst, energy)
        # compute the transmission probability from lead 0 to
        # lead 1
        data.append(smatrix.transmission(1, 0))
    return np.array(data)
