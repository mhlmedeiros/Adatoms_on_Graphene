import kwant
import tinyarray
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


#====================================================================#
#                    Helper classes                                  #
#====================================================================#
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

## Limited region with magnetic field:
class Bfield:
   def __init__(self, Bvalue, Length_centered):
       self.Bvalue = Bvalue
       self.L = Length_centered

   def __call__(self, x, y):
       return self.Bvalue if np.abs(x) <= self.L/2 else 0


## Hopping function's builder:
class HoppingFunction(object):

    def __init__(self, B_0_hopping, sign):
        self.B_0_hopping = B_0_hopping
        self.sign = sign

    def __call__(self, site1, site2, B, Lm, peierls):
        return self.sign * self.B_0_hopping * peierls(site1, site2, B, Lm)


#====================================================================#
#                    Lattice definitions                           #
#====================================================================#
## Matrix Definitions:
zeros_2x2 = tinyarray.array([[0,0],[0,0]])
sigma_0 = tinyarray.array([[1,0],[0,1]])
sigma_x = tinyarray.array([[0,1],[1,0]])
sigma_y = tinyarray.array([[0,-1j],[1j,0]])
sigma_z = tinyarray.array([[1,0],[0,-1]])


## Graphene lattice Definition:
sin_30 = 1/2
cos_30 = np.sqrt(3)/2

graphene = kwant.lattice.general([(1, 0), (sin_30, cos_30)], # primitive vectors
                                 [(0, 0), (0, 1 / np.sqrt(3))],        # coord. of basis atoms
                                 norbs = 2,                            # number of orbitals per site (spin)
                                 name='Graphene'                       # name of identification
                                )
## Split in sublattices:
A, B = graphene.sublattices

## Adatom lattice definition:
hydrogen = kwant.lattice.general([(1, 0), (sin_30, cos_30)],    # primitive vectors
                                 [(0, 0), (0, 1 / np.sqrt(3))], # coord. of basis atoms
                                 norbs = 2,
                                 name='H')

## Split in sublattices
HA, HB = hydrogen.sublattices

## Physical constants
g_Lande = 2
Magneton_Bohr = 5.788e-5 # eV/T


#====================================================================#
#                    Builder's helper functions                      #
#====================================================================#
## Zeeman:
def on_site_with_Zeeman(site, V, B, Lm):
    """
    This function defines the on-site energy by
    allowing to pass functions of position for
    the electrical potential and the magnetic field
    separately in order to account for the Zeeman
    effect.
    """
    x, y = site.pos
    Bfunc = Bfield(Bvalue=B, Length_centered=Lm)
    H_Z = g_Lande * Magneton_Bohr/2 * Bfunc(x,y) * sigma_z
    return V * sigma_0 + H_Z

def on_site_H_with_Zeeman(site, eps_H, B, Lm):
    """
    This function defines the on-site energy by
    allowing to pass functions of position for
    the electrical potential and the magnetic field
    separately in order to account for the Zeeman
    effect.
    """
    x, y = site.pos
    Bfunc = Bfield(Bvalue=B, Length_centered=Lm)
    H_Z = g_Lande * Magneton_Bohr/2 * Bfunc(x,y) * sigma_z
    return eps_H * sigma_0 + H_Z


## Peierls substitution:
def hopping_by_hand(Site1, Site2, t, B, Lm, peierls):
    return -t * sigma_0 * peierls(Site1, Site2, B, Lm)

def peierls_1(Site1, Site2, B, Lm):
    (x_i, y_i) = Site1.pos # Target
    (x_j, y_j) = Site2.pos # Source
    x_i, x_j = change_x(x_i, Lm), change_x(x_j, Lm)
    theta = B/2 * (x_i + x_j) * (y_i - y_j)
    return np.exp(-1j*theta)

def peierls_lead_L(Site1, Site2, B, Lm):
    (x_i, y_i) = Site1.pos # Target
    (x_j, y_j) = Site2.pos # Source
    theta = -B/2 * Lm * (y_i - y_j)
    return np.exp(-1j*theta)

def peierls_lead_R(Site1, Site2, B, Lm):
    (x_i, y_i) = Site1.pos # Target
    (x_j, y_j) = Site2.pos # Source
    theta = B/2 * Lm * (y_i - y_j)
    return np.exp(-1j*theta)

def change_x(x, Lm):
    if (-Lm/2) <= x <= (Lm/2): return x
    elif x > (Lm/2): return Lm/2
    else: return -Lm/2


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


#====================================================================#
#                          System Builders                           #
#====================================================================#
def make_graphene_strip(lattice, scatter_shape, t=1, iso=1e-6):

    syst = kwant.Builder()
    syst[lattice.shape(scatter_shape, (0, 0))] = on_site_with_Zeeman  # this is a func. of Bfield and pos.

    # Specify the hoppings for graphene lattice in the
    # format expected by builder.HoppingKind
    a, b = lattice.sublattices
    hoppings_list = (((0, 0), a, b), ((0, 1), a, b), ((-1, 1), a, b))

#     hopping = t * sigma_0

    syst[[kwant.builder.HoppingKind(*hop) for hop in hoppings_list]] = hopping_by_hand
    syst.eradicate_dangling()

    include_ISOC(syst, [a,b], lambda_I=iso)

    return syst

def make_graphene_leads(lattice, lead_shape, t=1, on_site=0, iso=1e-6):
    a, b = lattice.sublattices
    symmetry = kwant.TranslationalSymmetry((-1,0))
    symmetry.add_site_family(a, other_vectors=[(-1,2)])
    symmetry.add_site_family(b, other_vectors=[(-1,2)])

    lead_0 = kwant.Builder(symmetry)
    lead_0[lattice.shape(lead_shape, (0,0))] = zeros_2x2

    hoppings_list = (((0, 0), a, b), ((0, 1), a, b), ((-1, 1), a, b))
#     hopping = t * sigma_0
    lead_0[(kwant.builder.HoppingKind(*hop) for hop in hoppings_list)] = hopping_by_hand
    lead_0.eradicate_dangling()
    include_ISOC(lead_0, [a,b], lambda_I=iso)

    lead_1 = lead_0.reversed()

    lead_0 = lead_0.substituted(peierls='peierls_lead_L')
    lead_1 = lead_1.substituted(peierls='peierls_lead_R')

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

    H_isoc_matrix = -1j/3 * lambda_I/np.sqrt(3) * sigma_z # counter-clockwise

    H_isoc_p = HoppingFunction(B_0_hopping=H_isoc_matrix, sign=+1)
    H_isoc_m = HoppingFunction(B_0_hopping=H_isoc_matrix, sign=-1)

    system[kwant.builder.HoppingKind((1,0), sub_a, sub_a)]  = H_isoc_p
    system[kwant.builder.HoppingKind((0,1), sub_a, sub_a)]  = H_isoc_m
    system[kwant.builder.HoppingKind((-1,1), sub_a, sub_a)] = H_isoc_p
    system[kwant.builder.HoppingKind((1,0), sub_b, sub_b)]  = H_isoc_p
    system[kwant.builder.HoppingKind((0,1), sub_b, sub_b)]  = H_isoc_p
    system[kwant.builder.HoppingKind((-1,1), sub_b, sub_b)] = H_isoc_m

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
    H_asoc_matrix = family_id * 1j/3 * Lambda_I/np.sqrt(3) * sigma_z # clockwise
    H_asoc = HoppingFunction(B_0_hopping=H_asoc_matrix, sign=sign)

    system[CH_site, NNN_site] = H_asoc

def include_BR_sitewise(system, CH_site, NN_site, t=1, Lambda_BR=1):
    """
    Update the hopping between the CH_site and the NN_site to include
    the Bychkov-Rashba SOC.
    """
    # 1. Identify the hopping 2/2:
    dx, dy = np.sqrt(3) * (CH_site.pos - NN_site.pos)



    H_hop_matrix = t * sigma_0
    H_BR_matrix = (2j/3) * Lambda_BR * (dy * sigma_x - dx * sigma_y) ## (S X d_ij)_z

    H_matrix = H_hop_matrix + H_BR_matrix

    H_BR = HoppingFunction(B_0_hopping = H_matrix, sign=+1)


    # 2. Define the hopping
    system[CH_site, NN_site] = H_BR

def include_PIA_sitewise(system, site_target, site_source, lambda_I=1, Lambda_PIA=1):
    """
    Define the PIA hopping and add to the already existent hopping between
    site_target and site_source.
    """
    # 1.1 Identify the hopping 1/2:
    Dx, Dy = site_target.pos - site_source.pos

    # 1.2 Identify the hopping 2/2:
    delta_tag = site_target.tag - site_source.tag

    # 2. ISO
    sites_family = site_target.family
    if sites_family == A and delta_tag in ([0,1], [1,-1], [-1,0]):
        H_iso_matrix = 1j/3 * lambda_I/np.sqrt(3) * sigma_z
    elif sites_family == B and delta_tag in ([0,-1], [-1,+1], [1,0]):
        H_iso_matrix = 1j/3 * lambda_I/np.sqrt(3) * sigma_z
    else:
        H_iso_matrix = -1j/3 * lambda_I/np.sqrt(3) * sigma_z

    # 3. PIA:
    H_PIA_matrix =  2j/3 * Lambda_PIA * (Dy * sigma_x - Dx * sigma_y) ## (S x D_ij)_z

    # 4. Total hopping:
    H_PIA_ISO = HoppingFunction(B_0_hopping = (H_iso_matrix + H_PIA_matrix), sign=+1)

    system[site_target, site_source] = H_PIA_ISO

## Inserting adatom:
def insert_adatom(syst, pos_tag, sub_lat, t=1, l_iso=1, T=1, L_I=1, L_BR=1, L_PIA=1):

    if sub_lat == A:
        site_CH = A(*pos_tag)
        site_H = HA(*pos_tag)
    elif sub_lat == B:
        site_CH = B(*pos_tag)
        site_H = HB(*pos_tag)

    ## On-site:
    syst[site_H] = on_site_H_with_Zeeman
    ## Hopping:
    syst[site_H, site_CH] = T * sigma_0
    ## Neighbors
    nn_sites, nnn_sites = get_neighbors(syst, site_CH)

    ## Calculate and include the Adatom induced spin-orbit coupling
    for site in nnn_sites:
        include_ASO_sitewise(syst, site_CH, site, Lambda_I=L_I)

    ## Calculate and include into the system the Bychkov-Rashba spin-orbit coupling (BR)
    for site in nn_sites:
        include_BR_sitewise(syst, site_CH, site,t=t, Lambda_BR=L_BR)

    ## Calculate and include into the system the Pseudo-spin inversion asymmetry spin-orbit coupling (PIA)
    targets = [nn_sites[(i+1)%3] for i in range(3)]
    for site1, site2 in zip(targets, nn_sites):
        include_PIA_sitewise(syst, site1, site2, lambda_I=l_iso, Lambda_PIA=L_PIA)



#====================================================================#
#                          Plotting helpers                          #
#====================================================================#
def hopping_colors(site1, site2):
    if site1.family == site2.family:
        color='blue'
    else:
        color='black'
    return color

def hopping_lw(site1, site2):
    return 0.04 if site1.family == site2.family else 0.1

def family_colors(site):
        if site.family == A:
            color = 'k'
        elif site.family == B:
            color = 'w'
        elif site.family == HA:
            color = 'cyan'
        elif site.family == HB:
            color = 'cyan'
        else:
            color = 'cyan'
        return color


#====================================================================#
#                          Calculations                              #
#====================================================================#
## Conductance
def calculate_conductance(syst, energy_values, params_dict):
    syst = syst.finalized()
    data = []
    for energy in energy_values:
        # compute the scattering matrix at a given energy
        smatrix = kwant.smatrix(syst, energy, params=params_dict)
        # compute the transmission probability from lead 0 to
        # lead 1
        data.append(smatrix.transmission(1, 0))
    return np.array(data)

#====================================================================#
#                               Main                                 #
#====================================================================#
def main():
    ## Define the shape of the system:
    shape = Rectangle(W=5, L=5)

    ## Build the scattering region:
    system = make_graphene_strip(graphene, shape, t=1, iso=1e-6)

    ## Make the leads:
    leads  = make_graphene_leads(graphene, shape.leads, t=1, on_site=0, iso=1e-6)

    ## Attach the leads:
    for lead in leads:
        system.attach_lead(lead)

    pos_tag = (0,0)  # Adatom's tag
    sub_lat = A      # Adatom's Sublattice
    adatom_params = dict(t    = 2.6,
                         l_iso=12e-6,
                         T    = 7.5,
                         L_I  = -0.21e-3,
                         L_BR = 0.33e-3,
                         L_PIA= -0.77e-3)


    # In[23]:


    insert_adatom(system, pos_tag, sub_lat,  **adatom_params)


    # # Figure
    # fig, ax = plt.subplots(figsize=(20,5))
    # kwant.plot(system,
    #            site_color=family_colors,
    #            hop_color=hopping_colors,
    #            hop_lw=hopping_lw,
    #            site_lw=0.1, ax=ax)
    # ax.set_aspect('equal')
    # plt.show()

    # Calculate the transmission
    parameters_hand = dict(V=0,
                           t=1,
                           B=0.5*np.pi,
                           eps_H=0.16,
                           peierls=peierls_1,
                           peierls_lead_L=peierls_lead_L,
                           peierls_lead_R=peierls_lead_R,
                           Lm=3,
                        )
    energy_values = np.linspace(-0.5,0.5,300)
    transmission1 = calculate_conductance(system, energy_values, params_dict=parameters_hand)

    #
    plt.plot(energy_values, transmission1)
    # plt.plot(energy_values, transmission2)
    plt.show()


if __name__ == '__main__':
    main()
