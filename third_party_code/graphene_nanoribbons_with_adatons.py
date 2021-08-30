
"""
From:
https://wiki.physics.udel.edu/phys824/Conductance_of_topological_insulators_built_from_graphene_nanoribbons
"""

from math import sqrt
import random
import itertools as it
import tinyarray as ta
import numpy as np
import matplotlib.pyplot as plt
import kwant

class Honeycomb(kwant.lattice.Polyatomic):
    """Honeycomb lattice with methods for dealing with hexagons"""

    def __init__(self, name=''):
        prim_vecs = [[0.5, sqrt(3)/2], [1, 0]]  # bravais lattice vectors
        # offset the lattice so that it is symmetric around x and y axes
        basis_vecs = [[-0.5, -1/sqrt(12)], [-0.5, 1/sqrt(12)]]
        super(Honeycomb, self).__init__(prim_vecs, basis_vecs, name)
        self.a, self.b = self.sublattices

    def hexagon(self, tag):
        """ Get sites belonging to hexagon with the given tag.
            Returns sites in counter-clockwise order starting
            from the lower-left site.
        """
        tag = ta.array(tag)
        #         a-sites b-sites
        deltas = [(0, 0), (0, 0),
                  (1, 0), (0, 1),
                  (0, 1), (-1, 1)]
        lats = it.cycle(self.sublattices)
        return (lat(*(tag + delta)) for lat, delta in zip(lats, deltas))

    def hexagon_neighbors(self, tag, n=1):
        """ Get n'th nearest neighbor hoppings within the hexagon with
            the given tag.
        """
        hex_sites = list(self.hexagon(tag))
        return ((hex_sites[(i+n)%6], hex_sites[i%6]) for i in range(6))

def random_placement(builder, lattice, density):
    """ Randomly selects hexagon tags where adatoms can be placed.
        This avoids the edge case where adatoms would otherwise
        be placed on incomplete hexagons at the system boundaries.
    """
    assert 0 <= density <= 1
    system_sites = builder.sites()

    def hexagon_in_system(tag):
        return all(site in system_sites for site in lattice.hexagon(tag))

    # get set of tags for sites in system (i.e. tags from only one sublattice)
    system_tags = (s.tag for s in system_sites if s.family == lattice.a)
    # only allow tags which have complete hexagons in the system
    valid_tags = list(filter(hexagon_in_system, system_tags))
    N = int(density * len(valid_tags))
    total_hexagons=len(valid_tags)
    valuef=random.sample(valid_tags, N)
    return valuef

def ribbon(W, L):
    def shape(pos):
        return (-L <= pos[0] <= L and -W <= pos[1] <= W)
    return shape

## Pauli matrices ##
sigma_0 = ta.array([[1, 0], [0, 1]])  # identity
sigma_x = ta.array([[0, 1], [1, 0]])
sigma_y = ta.array([[0, -1j], [1j, 0]])
sigma_z = ta.array([[1, 0], [0, -1]])

## Hamiltonian value functions ##

def onsite_potential(site, params):
    return params['ep'] * sigma_0

def potential_shift(site, params):
    return params['mu'] * sigma_0

def kinetic(site_i, site_j, params):
    return -params['gamma'] * sigma_0

def rashba(site_i, site_j, params):
    d_ij = site_j.pos - site_i.pos
    rashba = 1j * params['V_R'] * (sigma_x * d_ij[1] - sigma_y * d_ij[0])
    return rashba + kinetic(site_i, site_j, params)

def spin_orbit(site_i, site_j, params):
    so = 1j * params['V_I'] * sigma_z
    return so

lat = Honeycomb()
pv1, pv2 = lat.prim_vecs
ysym = kwant.TranslationalSymmetry(pv2 - 2*pv1)  # lattice symmetry in -y direction
xsym = kwant.TranslationalSymmetry(-pv2)  # lattice symmetry in -x direction

# adatom lattice, for visualization only
adatom_lat = kwant.lattice.Monatomic(lat.prim_vecs, name='adatom')

def site_size(s):
    return 0.25 if s.family in lat.sublattices else 0.35

def site_color(s):
    return '#000000' if s.family in lat.sublattices else '#808080'

def create_lead_h(W, symmetry, axis=(0, 0)):
    lead = kwant.Builder(symmetry)
    lead[lat.wire(axis, W)] = 0. * sigma_0
    lead[lat.neighbors(1)] = kinetic
    return lead

def create_ribbon(W, L, adatom_density=0.2, show_adatoms=False):
    ## scattering region ##
    sys = kwant.Builder()
    sys[lat.shape(ribbon(W, L), (0, 0))] = onsite_potential
    sys[lat.neighbors(1)] = kinetic

    adatoms = random_placement(sys, lat, adatom_density)

    sys[(lat.hexagon(a) for a in adatoms)] = potential_shift
    sys[(lat.hexagon_neighbors(a, 1) for a in adatoms)] = rashba
    sys[(lat.hexagon_neighbors(a, 2) for a in adatoms)] = spin_orbit
    if show_adatoms:
        # no hoppings are added so these won't affect the dynamics; purely cosmetic
        sys[(adatom_lat(*a) for a in adatoms)] = 0

    ## leads ##
    leads = [create_lead_h(W, xsym)]
    leads += [lead.reversed() for lead in leads]  # right lead
    for lead in leads:
        sys.attach_lead(lead)
    return sys

def plot_ldos(sys, Es, params):
    fsys = sys.finalized()
    ldos = kwant.ldos(fsys, energy=Es, args=(params,))
    ldos = ldos[::2] + ldos[1::2] # sum spins
    kwant.plotter.map(fsys, ldos, vmax=0.1)

def plot_conductance(sys, energies,lead_i,lead_j, params):
    fsys = sys.finalized()
    data = []

    for energy in energies:
        smatrix = kwant.smatrix(fsys, energy, args=(params,))
        data.append(smatrix.transmission(lead_i,lead_j))

    plt.figure()
    plt.plot(energies, data)
    plt.xlabel("energy (t)")
    plt.ylabel("conductance (e^2/h)")
    plt.show()

if __name__ == '__main__':
#    params = dict(gamma=1., ep=0, mu=0., V_I=0.017, V_R=0)        # Tl adatoms
    params = dict(gamma=1., ep=0, mu=0, V_I=0.0032, V_R=0)         # In adatoms

    W=15
    L=30
    adatom_density=0.3

    sys = create_ribbon(W, L, adatom_density, show_adatoms=True)
    kwant.plot(sys, site_color=site_color, site_size=site_size)

    sys = create_ribbon(W, L, adatom_density, show_adatoms=False)
    plot_ldos(sys, 1e-5, params)

    Es = np.linspace(-1, 1, 100)
    plot_conductance(sys, Es, 1, 0, params)
