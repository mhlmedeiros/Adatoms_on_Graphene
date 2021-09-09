
"""
    Script to define a honeycomb lattice (Graphene) with periodic boundary
    conditions using Python-kwant. To keep things simple, we're going to
    consider hoppings between nearest-neighbors and disregard the spin degree
    of freedom.

    author: Marcos De Medeiros
    date: 09/08/2021
"""


import kwant
import numpy as np
import matplotlib.pyplot as plt

import scipy.linalg as la
import scipy.sparse.linalg as sla


## DEFINITIONS:
graphene = kwant.lattice.honeycomb(name='G')
a, b = graphene.sublattices

hydrogen = kwant.lattice.honeycomb(name='H')
ha, hb = hydrogen.sublattices


def make_system(N1=10, N2=20, pot=0, t=2.6):
    #### Define the scattering region. ####
    syst = kwant.Builder()
    for a1 in range(N1):
        for a2 in range(N2):
            syst[a(a1, a2)] = -pot
            syst[b(a1, a2)] = -pot

    # specify the hoppings of the graphene lattice in the
    # format expected by builder.HoppingKind
    hoppings = (((0, 0), a, b), ((0, 1), a, b), ((-1, 1), a, b))
    syst[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -t

    ## Horizontal hoppings PBC
    for j in range(1, N2):
        syst[a(N1-1, j), b(0,j-1)] = -t

    ## Vertical hopping PBC
    for i in range(N1):
        syst[b(i, N2-1), a(i, 0)] = -t
        syst[b((i+1) % N1, N2-1), a(i, 0)] = -t

    return syst


def insert_adatoms(system, adatom_concentration, n_cells_a1, n_cells_a2, verbatim=True):
    n_Carbon_sites = n_cells_a1 * n_cells_a2 * 2
    n_Hydrogen_sites = int(n_Carbon_sites // (100/adatom_concentration))

    if verbatim:
        print("Number of Carbon atoms = {:d}".format(n_Carbon_sites))
        print("Number of adatoms = {:d}".format(n_Hydrogen_sites))
        print("Adatoms' concentration = {:f} %".format(adatom_concentration))


    possible_tags_a1 = np.arange(n_cells_a1)
    possible_tags_a2 = np.arange(n_cells_a2)
    random_tags_a1 = np.random.choice(possible_tags_a1, n_Hydrogen_sites, replace=False)
    random_tags_a2 = np.random.choice(possible_tags_a2, n_Hydrogen_sites, replace=False)

    system = make_system(N1=n_cells_a1, N2=n_cells_a2)
    for i, j in zip(random_tags_a1, random_tags_a2):
        pos_adatom = (i, j)
        system = insert_single_adatom(system, pos_adatom)

    return system


def insert_single_adatom(system, pos_adatom, t=7.5, V=0.16):
    n1, n2 = pos_adatom
    system[ha(n1,n2)] = V
    system[ha(n1,n2), a(n1,n2)] = t
    return system


def dos_kpm(system, energies, additional_vectors=50, resolution=0.03):
    fsystem = system.finalized()
    spectrum = kwant.kpm.SpectralDensity(fsystem)
    spectrum.add_vectors(additional_vectors)
    spectrum.add_moments(energy_resolution=resolution)
    density_kpm = spectrum(energies)
    return density_kpm


def delta_approx(E, Em, eps):
    return (eps/np.pi)/((E-Em)**2 + eps**2)


def dos_manual(E_array, Eigenvalues, eps=0.1):
    dos = np.zeros_like(E_array)
    for Em in Eigenvalues:
        dos += delta_approx(E_array, Em, eps)
    return dos


def plot_density_of_states(E, DOS_pristine, DOS_adatoms):
    fig, ax = plt.subplots(ncols=2, figsize=(8,8))
    ax[0].tick_params(labelsize=20)
    ax[0].plot(E, np.real(DOS_pristine), label='Pristine', color='k', linestyle='--')
    ax[0].plot(E, np.real(DOS_adatoms), label='Adatoms', color='C2', linestyle='-')
    ax[0].legend(fontsize=20)
    ax[0].set_xlabel(r'$E$ [eV]',fontsize=24)
    ax[0].set_ylabel('DOS [a.u.]',fontsize=24)

    ax[1].tick_params(labelsize=20)
    ax[1].plot(E, np.real(DOS_adatoms)-np.real(DOS_pristine), label='Difference', color='C2', linestyle='-')
    ax[1].legend(fontsize=20)
    ax[1].set_xlabel(r'$E$ [eV]',fontsize=24)
    ax[1].set_yticks([])
    plt.tight_layout(pad=0.1)
    plt.show()



def main():
    n_cells_a1 = 200
    n_cells_a2 = 200
    adatom_concentration = 0.1  # [% of Carbon atoms]
    energies = np.linspace(-0.5, 0.5, 501)

    system = make_system(N1=n_cells_a1, N2=n_cells_a2, pot=0, t=2.6)
    # kwant.plot(system)

    density_pristine = dos_kpm(system, energies)

    system = insert_adatoms(system, adatom_concentration, n_cells_a1, n_cells_a2)
    density_adatoms = dos_kpm(system, energies)

    plot_density_of_states(energies, density_pristine, density_adatoms)




if __name__ == '__main__':
    main()
