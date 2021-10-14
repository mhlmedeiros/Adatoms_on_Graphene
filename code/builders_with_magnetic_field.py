import kwant
import tinyarray
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


#====================================================================#
#                    Lattice definitions                           #
#====================================================================#
## Matrix Definitions:
N_SPIN_DEGREE_OF_FREEDOM = 1
identity = np.identity(N_SPIN_DEGREE_OF_FREEDOM)

s_0 = np.array([[ 1,  0],[ 0, 1]])
s_x = np.array([[ 0,  1],[ 1, 0]])
s_y = np.array([[ 0,-1j],[1j, 0]])
s_z = np.array([[ 1,  0],[ 0,-1]])

H_J_matrix = tinyarray.array(np.kron(s_x, s_x) + np.kron(s_y, s_y) + np.kron(s_z, s_z))

sigma_0  = tinyarray.array(np.kron(s_0, identity))
sigma_x  = tinyarray.array(np.kron(s_x, identity))
sigma_y  = tinyarray.array(np.kron(s_y, identity))
sigma_z  = tinyarray.array(np.kron(s_z, identity))

## Graphene lattice Definition:
sin_30 = 1/2
cos_30 = np.sqrt(3)/2

graphene = kwant.lattice.general([(1, 0), (sin_30, cos_30)],           # primitive vectors
                                 [(0, 0), (0, 1 / np.sqrt(3))],        # coord. of basis atoms
                                 norbs = 2 * N_SPIN_DEGREE_OF_FREEDOM, # number of orbitals per site (spin)
                                 name='Graphene'                       # name of identification
                                )
## Split in sublattices:
A, B = graphene.sublattices

## Adatom lattice definition:
hydrogen = kwant.lattice.general([(1, 0), (sin_30, cos_30)],           # primitive vectors
                                 [(0, 0), (0, 1 / np.sqrt(3))],        # coord. of basis atoms
                                 norbs = 2 * N_SPIN_DEGREE_OF_FREEDOM, # number of orbitals per site (spin)
                                 name='H'                              # name of identification
                                 )

## Split in sublattices
HA, HB = hydrogen.sublattices

## Physical constants
g_Lande = 2
Magneton_Bohr = 5.788e-5 # eV/T


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
    def __init__(self, width, length, delta=2, centered=True, pbc=False):
        '''
        Calling the scattering region as strip:
        W = width of the strip
        L = length of the strip
        '''
        self.pbc = pbc
        if pbc: self.width, self.N = w_to_close_pbc(width)
        else: self.width, self.N = width, None
        self.length = length
        self.delta  = delta

        if centered and not pbc:
            self.x_sup =  length/2
            self.x_inf = -length/2
            self.y_sup =  width/2
            self.y_inf = -width/2
        else:
            self.x_inf = 0
            self.x_sup = self.length
            self.y_inf = 0
            self.y_sup = self.width


    def is_allowed(self, site):
        x, y = site.pos
        delta = self.delta
        x_min, x_max = self.x_inf + delta, self.x_sup - delta
        y_min, y_max = self.y_inf + delta, self.y_sup - delta
        return x_min <= x < x_max and y_min < y < y_max

    def leads(self, pos):
        _, y = pos
        y_inf, y_sup = self.y_inf, self.y_sup
        return y_inf < y <= y_sup

    def __call__(self, pos):
        x, y = pos
        x_inf, x_sup = self.x_inf, self.x_sup
        y_inf, y_sup = self.y_inf, self.y_sup
        return y_inf < y <= y_sup and x_inf <= x <= x_sup


## Limitted region with magnetic field:
class Bfield:
   def __init__(self, Bvalue, x_Binf=None, x_Bsup=None):
       """
       This class creates functions that return the value for the magnetic
       field depending on x-coordinate. The magnetic field returned is uniform
       inside a symmetrical interval

                x_Binf <= x <= x_Bsup

       and zero otherwise.

       If the length is not informed, the magnetic field will be constant and
       uniform for the whole system. The choice between these options has to
       be made alongside the choice of the gauge.
       """
       self.Bvalue = Bvalue
       self.x_Binf = x_Binf
       self.x_Bsup = x_Bsup

   def __call__(self, x, y):
        if self.x_Bsup:
            if self.x_Binf <= x <= self.x_Bsup:
                B_field = self.Bvalue
            else:
                B_field = 0
        else:
            B_field = self.Bvalue
        return self.Bvalue


## Hopping function's builder:
class HoppingFunction:
    def __init__(self, soc_hopping_matrix, simple_hopping_matrix=0, sign=+1):
        self.soc_hopping_matrix = soc_hopping_matrix
        self.simple_hopping_matrix = simple_hopping_matrix
        self.sign = sign

    def __call__(self, site1, site2, t, x_Binf, x_Bsup, peierls):
        H0 = t * self.simple_hopping_matrix
        HSOC = self.soc_hopping_matrix
        return self.sign * (H0 + HSOC) * peierls(site1, site2, B, x_Binf, x_Bsup)


class ISOHoppingFunction:
    def __init__(self, isoc_hopping_matrix, pia_hopping_matrix=0, sign=+1):
        self.isoc_hopping_matrix = isoc_hopping_matrix
        self.pia_hopping_matrix = pia_hopping_matrix
        self.sign = sign

    def __call__(self, site1, site2, lambda_iso, B, x_Binf, x_Bsup, peierls):
        H_ISO = lambda_iso * self.isoc_hopping_matrix
        H_PIA = self.pia_hopping_matrix
        return self.sign *  (H_ISO + H_PIA) * peierls(site1, site2, B, x_Binf, x_Bsup)


class OnSiteZeeman:
    def __init__(self, adatom_onsite=None, exchange=0):
        self.adatom_onsite = adatom_onsite
        self.exchange = exchange

    def __call__(self, site, B, x_Binf, x_Bsup, V):
        x, y = site.pos
        Bfunc = Bfield(Bvalue=B, x_Binf=x_Binf, x_Bsup=x_Bsup)
        H_Z = g_Lande * Magneton_Bohr/2 * Bfunc(x,y) * sigma_z
        H_J = self.exchange * H_J_matrix
        if self.adatom_onsite:
            onsite = self.adatom_onsite * sigma_0 + H_Z
        else:
            onsite = V * sigma_0 + H_Z
        if self.exchange and N_SPIN_DEGREE_OF_FREEDOM == 2:
            onsite = onsite - H_J
        return onsite


class SiteTest:
    def __init__(self, x, y):
        self.pos = (x, y)


# site_test = SiteTest(0,0)
# H_onsite = OnSiteZeeman(adatom_onsite=0.16, exchange=0.4)
# print(H_onsite(site_test, B=0, Lm=0, V=0))


def w_to_close_pbc(W):
    N = int(max(W // np.sqrt(3), 2))
    w_new = N * np.sqrt(3)
    return w_new, N


#====================================================================#
#                    Builder's helper functions                      #
#====================================================================#

## Peierls substitution:
def simple_hopping(Site1, Site2, t, B, x_Binf, x_Bsup, peierls):
    return -t * sigma_0 * peierls(Site1, Site2, B, x_Binf, x_Bsup)


def hopping_pbc(Site1, Site2, t, phi, B, x_Binf, x_Bsup):
    return -t * sigma_0 * np.exp(-1j*phi) * peierls_pbc(Site1, Site2, B, x_Binf, x_Bsup)


def peierls_scatter(Site1, Site2, B, x_Binf, x_Bsup):
    """
    This phase factor correspond to the gauge where the magnetic
    field is limitted to a interval inside the scattering region.
    """
    (x_i, y_i) = Site1.pos # Target
    (x_j, y_j) = Site2.pos # Source
    if x_Bsup:
        x_i, x_j = change_x(x_i, x_Binf, x_Bsup), change_x(x_j, x_Binf, x_Bsup)
        theta = B/2 * (x_i + x_j) * (y_i - y_j)
    else:
        theta = -B/2 * (x_i - x_j) * (y_i + y_j)
    return np.exp(2j*np.pi*theta)


def peierls_pbc(Site1, Site2, B, x_Binf, x_Bsup):
    """
    This phase factor correspond to the gauge where the magnetic
    field is limitted to a interval inside the scattering region.
    """
    (x_i, y_i) = Site1.pos # Target
    (x_j, y_j) = Site2.pos # Source
    x_i, x_j = change_x(x_i, x_Binf, x_Bsup), change_x(x_j, x_Binf, x_Bsup)
    theta = B/2 * (x_i + x_j) * 1/np.sqrt(3)
    return np.exp(2j*np.pi*theta)


def peierls_lead_L(Site1, Site2, B, x_Binf, x_Bsup):
    """
    When 'peierls_scatter' is used, this has to be the
    the phase factor for left-hand lead.
    """
    (x_i, y_i) = Site1.pos # Target
    (x_j, y_j) = Site2.pos # Source

    if x_Binf:
        theta = B/2 * (2 * x_Binf) * (y_i - y_j)
    else:
        theta = -B/2 * (x_i - x_j) * (y_i + y_j)

    return np.exp(2j*np.pi*theta)


def peierls_lead_R(Site1, Site2, B, x_Binf, x_Bsup):
    """
    When 'peierls_scatter' is used, this has to be the
    the phase factor for right-hand lead.
    """
    (x_i, y_i) = Site1.pos # Target
    (x_j, y_j) = Site2.pos # Source

    if x_Bsup:
        theta = B/2 * (2 * x_Bsup) * (y_i - y_j)
    else:
        theta = -B/2 * (x_i - x_j) * (y_i + y_j)

    return np.exp(-2j*np.pi*theta)


def change_x(x, x_Binf, x_Bsup):
    # if Lm:
    if x_Binf <= x <= x_Bsup:
        x_transformed = x
    elif x > x_Bsup:
        x_transformed = x_Bsup
    else:
        x_transformed = x_Binf
    # else:
    #     x_transformed = x
    return x_transformed


## Getting the neighboring sites
def get_neighbors(system, C_H_site):
    """
    Returns a list containing 2 other lists:
        - nn_list = list of nearest neighbors sites
        - nnn_list = list of next nearest neighbors sites
    """
    site_tag = C_H_site.tag
    site_sub = C_H_site.family
    nn_list = get_nn(system, site_tag, site_sub)
    nnn_list = get_nnn(system, site_tag, site_sub)
    return [nn_list, nnn_list]


def get_nn(system, tag, sub_s):
    """
    system := kwant.builder.Builder
    tag    := tuple (i,j) of the site's tag
    sub_s  := site's sublattice

    Notice that
    """
    list_sub_lat = [A, B]
    list_sub_lat.remove(sub_s) # remove the sublattice to which the site belongs
    sub_nn, = list_sub_lat     # extract the sublattice of the neighbors
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


def get_adatom_AB_positions(total_number_of_adatoms, allowed_sites):
    A_adatoms_to_place = total_number_of_adatoms // 2
    B_adatoms_to_place = total_number_of_adatoms - A_adatoms_to_place
    list_of_A_adatoms = []
    list_of_B_adatoms = []

    while A_adatoms_to_place or B_adatoms_to_place:
        site_adatom = allowed_sites[np.random.choice(len(allowed_sites))]
        if site_adatom.family == A and A_adatoms_to_place:
            list_of_A_adatoms.append(site_adatom)
            allowed_sites = exclude_neighboring_sites(site_adatom, allowed_sites)
            A_adatoms_to_place -= 1
        elif site_adatom.family == B and B_adatoms_to_place:
            list_of_B_adatoms.append(site_adatom)
            allowed_sites = exclude_neighboring_sites(site_adatom, allowed_sites)
            B_adatoms_to_place -= 1
        if not allowed_sites:
            break
    return list_of_A_adatoms, list_of_B_adatoms, allowed_sites


def exclude_neighboring_sites(adatom_site, list_of_sites, radius=2):
    sites_to_exclude = neighboring_sites(adatom_site, list_of_sites, radius)
    for site in sites_to_exclude:
        list_of_sites.remove(site)
    return list_of_sites


def neighboring_sites(adatom_site, list_of_sites, radius):
    list_of_neighboring_sites = []
    xA, yA = adatom_site.pos
    for site in list_of_sites:
        x, y = site.pos
        if (x-xA)**2 + (y-yA)**2 <= radius**2:
            list_of_neighboring_sites.append(site)
    return list_of_neighboring_sites



# TODO: UPDATE THE REST OF THE CODE STARTING HERE (14/09/2021)
#====================================================================#
#                          System Builders                           #
#====================================================================#
def make_graphene_strip(lattice, shape):

    syst = kwant.Builder()
    on_site_carbon = OnSiteZeeman()
    syst[lattice.shape(shape, (0, 0))] = on_site_carbon  # this is a func. of Bfield and pos.

    # Specify the hoppings for graphene lattice in the
    # format expected by builder.HoppingKind
    A, B = lattice.sublattices
    hoppings_list = (((0, 0), A, B), ((0, 1), A, B), ((-1, 1), A, B))

    syst[[kwant.builder.HoppingKind(*hop) for hop in hoppings_list]] = simple_hopping

    if shape.pbc:
        sites_x_tags = [s.tag[0] for s in syst.sites()
                       if (s.family == B and s.tag[1]==0)]
        N = shape.N
        M = max(sites_x_tags) + 1
        for i in range(M):
            syst[A(i-N, 2*N), B(i, 0)] = hopping_pbc
    else:
        syst.eradicate_dangling()
        include_ISOC(syst, [A,B])

    return syst


def make_graphene_leads(lattice, shape):
    A, B = lattice.sublattices
    symmetry = kwant.TranslationalSymmetry((-1,0))
    symmetry.add_site_family(A, other_vectors=[(-1,2)])
    symmetry.add_site_family(B, other_vectors=[(-1,2)])

    if N_SPIN_DEGREE_OF_FREEDOM == 1 :
        lead_0 = kwant.Builder(symmetry, conservation_law=-sigma_z)
    else:
        sigma_tilde = np.diag(np.linspace(1, 4, 4)) # DIAG([1, 2, 3, 4])
        lead_0 = kwant.Builder(symmetry, conservation_law=sigma_tilde)

    # On-site energy is the same of the scattering region (by now)
    on_site_carbon = OnSiteZeeman()
    lead_0[lattice.shape(shape.leads, (0,0))] = on_site_carbon

    hoppings_list = (((0, 0), A, B), ((0, 1), A, B), ((-1, 1), A, B))
    lead_0[(kwant.builder.HoppingKind(*hop) for hop in hoppings_list)] = simple_hopping

    if shape.pbc:
        N = shape.N
        lead_0[A(-N, 2*N), B(0, 0)] = hopping_pbc
    else:
        lead_0.eradicate_dangling()
        include_ISOC(lead_0, [A,B])

    lead_1 = lead_0.reversed()
    lead_0 = lead_0.substituted(peierls='peierls_lead_L')
    lead_1 = lead_1.substituted(peierls='peierls_lead_R')

    return [lead_0, lead_1]


def include_ISOC(system, G_sub_lattices):
    """
    ## INCLUDING THE INTRINSIC SOC (isoc):
    system         := kwant.builder.Builder
    G_sub_lattices := list of Graphene sublattices
    """
    sub_a, sub_b = G_sub_lattices


    H_isoc_matrix = -1j/3 * 1/np.sqrt(3) * sigma_z # counter-clockwise

    H_isoc_p = ISOHoppingFunction(isoc_hopping_matrix=H_isoc_matrix, sign=+1)
    H_isoc_m = ISOHoppingFunction(isoc_hopping_matrix=H_isoc_matrix, sign=-1)

    system[kwant.builder.HoppingKind((1,0), sub_a, sub_a)]  = H_isoc_p
    system[kwant.builder.HoppingKind((0,1), sub_a, sub_a)]  = H_isoc_m
    system[kwant.builder.HoppingKind((-1,1), sub_a, sub_a)] = H_isoc_p
    system[kwant.builder.HoppingKind((1,0), sub_b, sub_b)]  = H_isoc_p
    system[kwant.builder.HoppingKind((0,1), sub_b, sub_b)]  = H_isoc_p
    system[kwant.builder.HoppingKind((-1,1), sub_b, sub_b)] = H_isoc_m


## INSERTING ADATOMS:
def insert_adatom(syst, pos_tag, sub_lat, eps_H=1, T=1, L_I=1, L_BR=1, L_PIA=1):

    if sub_lat == A:
        site_CH = A(*pos_tag)
        site_H = HA(*pos_tag)
    elif sub_lat == B:
        site_CH = B(*pos_tag)
        site_H = HB(*pos_tag)

    ## On-site:
    on_site_adatom = OnSiteZeeman(adatom_onsite=eps_H)
    syst[site_H] = on_site_adatom
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


def insert_adatoms_randomly(system, shape, adatom_concentration, adatom_params):
    """
    It is the almost the same function defined in "Periodic_Boundary_Conditions.py",
    located in this same directory.
    """
    ## INSERTING ADATOMS:
    print("Collecting the allowed sites to place adatoms...", end=' ')
    allowed_sites = [site for site in system.sites() if shape.is_allowed(site)]
    print("OK")

    print("Inserting adatoms randomly...", end=' ')
    n_Carbon_sites = len(system.sites())
    n_Hydrogen_sites = int(n_Carbon_sites // (100/adatom_concentration))
    number_of_adatoms = max(1, n_Hydrogen_sites)
    list_of_A_sites, list_of_B_sites, allowed_sites = get_adatom_AB_positions(
                                                            number_of_adatoms,
                                                            allowed_sites)
    A_adatoms_tags = [site.tag for site in list_of_A_sites]
    B_adatoms_tags = [site.tag for site in list_of_B_sites]
    CH_sites = list_of_A_sites + list_of_B_sites

    all_neighbors = [get_neighbors(system, CH) for CH in CH_sites]
    all_NN_neighbors = [A[0] for A in all_neighbors]
    all_NNN_neighbors = [A[1] for A in all_neighbors]

    T = adatom_params['T']          #  5.5 eV Fluorine,  7.5 eV Hydrogen: ADATOM-CARBON HOPPING
    epsilon = adatom_params['eps']  # -2.2 eV Fluorine, 0.16 eV Hydrogen: ON-SITE ENERGY FOR ADATOM
    J_exchange = adatom_params['exchange']
    on_site_adatom = OnSiteZeeman(adatom_onsite=epsilon, exchange=J_exchange)

    for tagA in A_adatoms_tags:
        system[HA(*tagA)] = on_site_adatom     # on-site
        system[A(*tagA), HA(*tagA)] = sigma_0 * T # hopping with C_H

    for tagB in B_adatoms_tags:
        system[HB(*tagB)] = on_site_adatom     # on-site
        system[B(*tagB), HB(*tagB)] = sigma_0 * T # hopping with C_H

    print("OK")

    print("Considering the SOC terms...", end=" ")
    ## Calculate and include the Adatom induced spin-orbit coupling (ASO)
    include_all_ASO(system, CH_sites, all_NNN_neighbors, Lambda_I=adatom_params['Lambda_I'])

    ## Calculate and include into the system the Bychkov-Rashba spin-orbit coupling (BR)
    include_all_BR(system, CH_sites, all_NN_neighbors, Lambda_BR=adatom_params['Lambda_BR'])

    ## Calculate and include into the system the Pseudo-spin inversion asymmetry spin-orbit coupling (PIA)
    include_all_PIA(system, all_NN_neighbors, Lambda_PIA=adatom_params['Lambda_PIA'])
    print("OK")
    # print("Formating sites for plotting...", end=' ')
    # format_sites_3 = FormatMapSites(allowed_sites, CH_sites)
    # print("OK")
    return system


# FOR ALL ADATOMS:
def include_all_ASO(system, all_CH_sites, all_NNN_neighbors, Lambda_I=1):
    for CH_site, NNN_sites in zip(all_CH_sites, all_NNN_neighbors):
        for NNN_site in NNN_sites:
            include_ASO_sitewise(system, CH_site, NNN_site, Lambda_I)


def include_all_BR(system, all_CH_sites, all_NN_neighbors, Lambda_BR=1):
    for CH_site, NN_sites in zip(all_CH_sites, all_NN_neighbors):
        for NN_site in NN_sites:
            include_BR_sitewise(system, CH_site, NN_site, Lambda_BR)


def include_all_PIA(system, all_NN_neighbors, Lambda_PIA=1):
    for NN_sites in all_NN_neighbors:
        targets = [NN_sites[(i+1)%3] for i in range(3)]
        for site1, site2 in zip(targets, NN_sites):
            include_PIA_sitewise(system, site1, site2, Lambda_PIA)


# FOR EVERY ADATOM
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
    H_asoc = HoppingFunction(soc_hopping_matrix=H_asoc_matrix, sign=sign)

    system[CH_site, NNN_site] = H_asoc


def include_BR_sitewise(system, CH_site, NN_site, Lambda_BR=1):
    """
    Update the hopping between the CH_site and the NN_site to include
    the Bychkov-Rashba SOC.
    """
    # 1. Identify the hopping:
    dx, dy = np.sqrt(3) * (CH_site.pos - NN_site.pos)
    # 2. Calculate the hopping:
    H_BR_matrix = (2j/3) * Lambda_BR * (dy * sigma_x - dx * sigma_y) ## (S X d_ij)_z
    H_BR = HoppingFunction(soc_hopping_matrix = H_BR_matrix, simple_hopping_matrix = sigma_0, sign=+1)
    # 3. (Re)Define the hopping including BR-SOC
    system[CH_site, NN_site] = H_BR


def include_PIA_sitewise(system, site_target, site_source, Lambda_PIA=1):
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
        H_iso_matrix = 1j/3 * 1/np.sqrt(3) * sigma_z
    elif sites_family == B and delta_tag in ([0,-1], [-1,+1], [1,0]):
        H_iso_matrix = 1j/3 * 1/np.sqrt(3) * sigma_z
    else:
        H_iso_matrix = -1j/3 * 1/np.sqrt(3) * sigma_z

    # 3. PIA:
    H_PIA_matrix =  2j/3 * Lambda_PIA * (Dy * sigma_x - Dx * sigma_y) ## (S x D_ij)_z

    # 4. Total hopping:
    H_PIA_ISO = ISOHoppingFunction(isoc_hopping_matrix = H_iso_matrix, pia_hopping_matrix = H_PIA_matrix, sign=+1)

    system[site_target, site_source] = H_PIA_ISO


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
    shape = Rectangle(width=5, length=5, delta=1, pbc=True)

    ## Build the scattering region:
    system = make_graphene_strip(graphene, shape)

    ## Make the leads:
    leads  = make_graphene_leads(graphene, shape)

    ## Attach the leads:
    for lead in leads:
        system.attach_lead(lead)

    pos_tag = (1,2)  # Adatom's tag
    sub_lat = A      # Adatom's Sublattice
    adatom_params = dict(T     = 7.5,
                         eps_H = 0.16,
                         L_I   = -0.21e-3,
                         L_BR  = 0.33e-3,
                         L_PIA = -0.77e-3)
    # insert_adatom(system, pos_tag, sub_lat,  **adatom_params)


    # Figure
    # fig, ax = plt.subplots(figsize=(20,5))
    # kwant.plot(system,
    #            site_color=family_colors,
    #            hop_color=hopping_colors,
    #            hop_lw=hopping_lw,
    #            site_lw=0.1, ax=ax)
    # ax.set_aspect('equal')
    # plt.show()

    # Calculate the transmission
    Bflux = 0.     # in units of quantum of magnetic flux
    Bfield = Bflux / (np.sqrt(3)/2) # sqrt(3)/2 == hexagon area
    parameters_hand = dict(V = 0,
                           t = 2.6,
                           phi = 0,
                           lambda_iso = 12e-6,
                           B = Bfield,
                           x_Binf = 1,
                           x_Bsup = 4,
                           peierls = peierls_scatter,
                           peierls_lead_L = peierls_lead_L,
                           peierls_lead_R = peierls_lead_R)
    # energy_values = np.linspace(-5, 5, 300)
    # transmission1 = calculate_conductance(system, energy_values,
    #                                      params_dict=parameters_hand)
    #
    # # PLOT THE CONDUCTION
    # plt.plot(energy_values, transmission1)
    # # plt.plot(energy_values, transmission2)
    # plt.show()

    smatrix = kwant.smatrix(system.finalized(), 0.4, params=parameters_hand)
    modes_per_spin = smatrix.num_propagating(0)/4 ## DIVIDING ONLY AMONG ELECTRONIC SPINS
    # modes_per_spin = 1

    ## THE EXTRA DIVISION BY 2 REQUIRES CLARIFICATION, BUT IT HAS TO DO WITH THE DENSITY MATRIX
    ## OF AN UNPOLARIZED SYSTEM \rho =  1/2 |UP)(UP| + 1/2 |DN)(DN|
    ## ADOPTED WHEN TRANCING OUT THE IMPURITY SPIN DEGREE OF FREEDOM:
    # T_up_up = np.linalg.norm(1/2 * (smatrix.submatrix((1,0), (0,0)) + smatrix.submatrix((1,1), (0,1))))**2 * 1/modes_per_spin
    # T_up_dn = np.linalg.norm(smatrix.submatrix((1,0), (0,2)) + smatrix.submatrix((1,1), (0,3)))**2/2/modes_per_spin
    # T_dn_up = np.linalg.norm(smatrix.submatrix((1,2), (0,0)) + smatrix.submatrix((1,3), (0,1)))**2/2/modes_per_spin
    # T_dn_dn = np.linalg.norm(smatrix.submatrix((1,2), (0,2)) + smatrix.submatrix((1,3), (0,3)))**2/2/modes_per_spin
    # print("\nT_sz = \n[[{:.3f}, {:.3f}],\n[{:.3f}, {:.3f}]]".format(T_up_up, T_up_dn, T_dn_up, T_dn_dn))
    #
    # R_up_up = np.linalg.norm(smatrix.submatrix((0,0), (0,0)) + smatrix.submatrix((0,1), (0,1)))**2/2/modes_per_spin
    # R_up_dn = np.linalg.norm(smatrix.submatrix((0,0), (0,2)) + smatrix.submatrix((0,1), (0,3)))**2/2/modes_per_spin
    # R_dn_up = np.linalg.norm(smatrix.submatrix((0,2), (0,0)) + smatrix.submatrix((0,3), (0,1)))**2/2/modes_per_spin
    # R_dn_dn = np.linalg.norm(smatrix.submatrix((0,2), (0,2)) + smatrix.submatrix((0,3), (0,3)))**2/2/modes_per_spin
    # print("\nR_sz = \n[[{:.3f}, {:.3f}],\n[{:.3f}, {:.3f}]]".format(R_up_up, R_up_dn, R_dn_up, R_dn_dn))
    #
    # total_transmission = smatrix.transmission(1,0)
    # total_reflection = smatrix.transmission(0,0)
    # propagating_modes = 1 #smatrix.num_propagating(0)/4

    # print(f"\nNumber of propagating modes = {smatrix.num_propagating(0)}")
    # print(f"Total transmission = {total_transmission}")
    # print(f"Total reflection = {total_reflection}")
    # print(f"T + R = {total_transmission + total_reflection}\n")
    #
    #
    # T_upUp_upUp = smatrix.transmission((1,0),(0,0))/propagating_modes
    # R_upUp_upUp = smatrix.transmission((0,0),(0,0))/propagating_modes
    #
    # T_upDn_upDn = smatrix.transmission((1,1),(0,1))/propagating_modes
    # R_upDn_upDn = smatrix.transmission((0,1),(0,1))/propagating_modes
    #
    # T_dnUp_dnUp = smatrix.transmission((1,2),(0,2))/propagating_modes
    # R_dnUp_dnUp = smatrix.transmission((0,2),(0,2))/propagating_modes
    #
    # T_dnDn_dnDn = smatrix.transmission((1,3),(0,3))/propagating_modes
    # R_dnDn_dnDn = smatrix.transmission((0,3),(0,3))/propagating_modes
    #
    # print("T_upUp_upUp = {:.3f}\nR_upUp_upUp = {:.3f}\n".format(T_upUp_upUp, R_upUp_upUp))
    # print("T_upUp_upUp + R_upUp_upUp = {:.3f}\n".format(T_upUp_upUp + R_upUp_upUp))
    #
    # print("T_upDn_upDn = {:.3f}\nR_upDn_upDn = {:.3f}\n".format(T_upDn_upDn, R_upDn_upDn))
    # print("T_upDn_upDn + R_upDn_upDn = {:.3f}\n".format(T_upDn_upDn + R_upDn_upDn))
    #
    # print("T_dnUp_dnUp = {:.3f}\nR_dnUp_dnUp = {:.3f}\n".format(T_dnUp_dnUp, R_dnUp_dnUp))
    # print("T_dnUp_dnUp + R_dnUp_upUp = {:.3f}\n".format(T_dnUp_dnUp + R_dnUp_dnUp))
    #
    # print("T_dnDn_dnDn = {:.3f}\nR_dnDn_dnDn = {:.3f}\n".format(T_dnDn_dnDn, R_dnDn_dnDn))
    # print("T_dnDn_upDn + R_dnDn_upDn = {:.3f}\n".format(T_dnDn_dnDn + R_dnDn_dnDn))


if __name__ == '__main__':
    main()
