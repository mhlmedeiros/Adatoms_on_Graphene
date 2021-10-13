import kwant
import tinyarray
import numpy as np
import numpy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.constants as sci
from itertools import product
import builders_with_magnetic_field as bm
import graphene_strip_zigzag as gz


HBAR  = sci.physical_constants['reduced Planck constant in eV s'][0]


def calculate_relaxation_time(system, energies, syst_params, n_phases=20):
    """
    Calculate the inverse of spin relaxation time for each value of energy given.

    :param system: kwant.Builder
    :param energies: numpy.Array
    :param syst_params: dict
    :param n_phases: int

    :return tau_inv: numpy.Array
    """

    CONST = 4 * syst_params['t']/HBAR * syst_params['eta'] * syst_params['width']
    tau_inv = np.empty_like(energies)
    phi_values = np.linspace(0, 2*np.pi, n_phases)


    for i, energy in enumerate(energies):
        print("Calculating the smatrix for E = {:2.2f} ... ".format(energy), end='')
        gamma_sz = 0
        n_phases_with_propag = 0

        for phi in phi_values:
            syst_params["phi"] = phi
            smatrix = kwant.smatrix(system.finalized(), energy, params=syst_params)
            g_sz, modes_per_spin = spin_flip_probability(smatrix)
            gamma_sz += g_sz

            if modes_per_spin:
                n_phases_with_propag += 1

        n_phases_with_propag = max(n_phases_with_propag, 1)

        tau_inv[i] = CONST * gamma_sz/n_phases_with_propag
        print("OK")
    return tau_inv


def spin_flip_probability(smatrix):
    """
    This function returns the spin-flip probability per mode for
    out-of-plane spins:
        * gamma_sz
    and the number of propagating modes:
        * modes_per_spin

    When modes_per_spin == 0, the function returns two zeros.
    """

    modes_per_spin = smatrix.num_propagating(0)/4
    if modes_per_spin:
        t_sz_list, r_sz_list = calculate_sz_matrices(smatrix)
        gamma_sz = (la.norm(t_sz_list[1])**2 + la.norm(t_sz_list[2])**2
                 + la.norm(r_sz_list[1])**2 + la.norm(r_sz_list[2])**2)/modes_per_spin
        print("\tgamma_sz = ",gamma_sz)
    else:
        print("\tmodes per spin = ", modes_per_spin)
        gamma_sz = 0
    return gamma_sz, modes_per_spin


def calculate_sz_matrices(smatrix):
    """
    This function returns two 2x2 matrices that contain the
    transmission and reflection resolved by spin projections:

    t_matrix = [[T_up_up, T_up_dn],
                [T_dn_up, T_dn_dn]]

    r_matrix = [[R_up_up, R_up_dn],
                [R_dn_up, R_dn_dn]]

    The WHOLE matrices are important because with then it is
    possible to get the correspondent matrices for in-plane
    spin projections.

    """
    t_source, t_target = 0, 1
    r_source, r_target = 0, 0
    tuu, tud, tdu, tdd = transmission_matrices(smatrix, t_source, t_target)
    ruu, rud, rdu, rdd = transmission_matrices(smatrix, r_source, r_target)

    transmission_list = [tuu, tud, tdu, tdd]
    reflection_list = [ruu, rud, rdu, rdd]

    return transmission_list, reflection_list


def transmission_matrices(smatrix, lead_source, lead_target):
    """
    Returns the partial trace for the smatrix, considering the density matrix
    for the impurity spin defined by:

        Rho_imp = 1/2 {|Up)(Up| + |Down)(Down|}

    """
    ## EXISTE UM PROBLEMA AQUI!!!
    ## PRECISAMOS CONSIDERAR O FLIP DE AMBOS OS SPINS
    ## ATUALMENTE, COM O TRAÇO PARCIAL SOBRE O ESPAÇO DO SPIN DA IMUREZA
    ## PERDE-SE A INFORMAÇÃO DESSE FLIP.
    s_up_up     = 1/2 * (smatrix.submatrix((lead_target, 0), (lead_source, 0))
                  + smatrix.submatrix((lead_target, 1), (lead_source, 1)))

    s_up_down   = 1/2 * (smatrix.submatrix((lead_target, 0), (lead_source, 2)) +
                  + smatrix.submatrix((lead_target, 1), (lead_source, 3)))

    s_down_up   = 1/2 * (smatrix.submatrix((lead_target, 2), (lead_source, 0))
                  + smatrix.submatrix((lead_target, 3), (lead_source, 1)))

    s_down_down = 1/2 * (smatrix.submatrix((lead_target, 2), (lead_source, 2))
                  + smatrix.submatrix((lead_target, 3), (lead_source, 3)))

    return s_up_up, s_up_down, s_down_up, s_down_down


################################################################################
#                  FUNCTIONS TO CONSIDER BOTH SPIN-FLIPS:                      #
################################################################################

def spin_relaxation_imp(system, energies, syst_params, n_phases=20):
    CONST = 4 * syst_params['t']/HBAR * syst_params['eta'] * syst_params['width']
    tau_inv = np.empty_like(energies)
    phi_values = np.linspace(0, 2*np.pi, n_phases)


    for i, energy in enumerate(energies):
        print("Calculating the smatrix for E = {:2.2f} ... ".format(energy), end='')
        gamma_sz = 0
        n_phases_with_propag = 0

        for phi in phi_values:
            syst_params["phi"] = phi
            smatrix = kwant.smatrix(system.finalized(), energy, params=syst_params)
            g_sz, modes_per_spin = spin_flip_probability_imp(smatrix) ## the sonly diff from 'calculate_relaxation_time'
            gamma_sz += g_sz

            if modes_per_spin:
                n_phases_with_propag += 1

        n_phases_with_propag = max(n_phases_with_propag, 1)

        tau_inv[i] = CONST * gamma_sz/n_phases_with_propag
        print("OK")
    return tau_inv


def spin_flip_probability_imp(smatrix):
    modes_per_spin = 2* smatrix.num_propagating(0)/4
    if modes_per_spin:
        transmission = scattering_amplitude_imp(smatrix, 0, 1)
        reflection = scattering_amplitude_imp(smatrix, 0, 0)
        probability = 1/modes_per_spin * (la.norm(transmission)**2 + la.norm(reflection)**2)
    else:
        probability = 0
    return probability, modes_per_spin


def scattering_amplitude_imp(smatrix, lead_source, lead_target):
    """
    Calculate the scattering amplitude for both electron and impurity
    spin to flip.
    """
    return 1/2 *(smatrix.submatrix((lead_target, 1), (lead_source, 2)) +
                 smatrix.submatrix((lead_target, 2), (lead_source, 1)))


################################################################################
#                          FUNCTIONS TO PERFORM TESTS:                         #
################################################################################
def test_matrices(system, energy, params):
    smatrix = kwant.smatrix(system.finalized(), energy, params=params)
    modes_per_spin = smatrix.num_propagating(0)/4
    t_sz_list, r_sz_list = calculate_sz_matrices(smatrix)
    show_matrices(t_sz_list, r_sz_list, modes_per_spin, spin='z')


def show_matrices(t_list, r_list, modes_per_spin, spin='z'):
    T_prob = [la.norm(T)**2/modes_per_spin for T in t_list]
    R_prob = [la.norm(R)**2/modes_per_spin for R in r_list]
    print("\nTRANSMISSION S{}: ".format(spin))
    print("[[{:.3f}, {:.3f}],\n [{:.3f}, {:.3f}]]".format(*T_prob))
    print("\nREFLECTION S{}: ".format(spin))
    print("[[{:.3f}, {:.3f}],\n [{:.3f}, {:.3f}]]".format(*R_prob))



def main():
    ## Define the shape of the system: width=130, length=62, pbc=True
    shape = bm.Rectangle(width=130, length=62, pbc=True)

    ## Build the scattering region:
    system = bm.make_graphene_strip(bm.graphene, shape)

    ## Make the leads:
    leads  = bm.make_graphene_leads(bm.graphene, shape)

    ## Attach the leads:
    for lead in leads:
        system.attach_lead(lead)


    ## INSERT THE ADATOM:
    number_of_carbons =  len(system.sites())
    eta = 1/number_of_carbons
    density_ppm = eta * 1e6 # [ppm]
    density_percent = density_ppm * 1e-4

    L_I   = -0.21e-3  ## [eV] INTRINSIC SOC
    L_BR  =  0.33e-3  ## [eV] RASHBA SOC
    L_PIA = -0.77e-3  ## [eV] PSEUDO INVERSION ASYMMETRY
    J_exchange = -0.4 ## [eV] Exchange

    H_params = dict(T = 7.5, eps = 0.16, Lambda_I = L_I, Lambda_BR = L_BR, Lambda_PIA = L_PIA, exchange = J_exchange)
    H_params_with_J = dict(T = 7.5, eps = 0.16, Lambda_I = 0, Lambda_BR = 0, Lambda_PIA = 0, exchange = J_exchange)
    H_params_only_ISO = dict(T = 7.5, eps = 0.16, Lambda_I = L_I, Lambda_BR =    0, Lambda_PIA =     0, exchange = J_exchange)
    H_params_only_BR  = dict(T = 7.5, eps = 0.16, Lambda_I =   0, Lambda_BR = L_BR, Lambda_PIA =     0, exchange = J_exchange)
    H_params_only_PIA = dict(T = 7.5, eps = 0.16, Lambda_I =   0, Lambda_BR =    0, Lambda_PIA = L_PIA, exchange = J_exchange)
    system = bm.insert_adatoms_randomly(system, shape, density_percent, H_params_with_J)


    # CALCULATE THE TRANSMISSION COEFFICIENTS
    Bfield = 0
    syst_params = dict(
                    width=shape.width, # system width
                    eta=eta,           # density of impurities
                    V=0,               # on-site C-atoms
                    t=2.6,             # hoppings C-atoms
                    phi=0,             # PBC hopping phase
                    lambda_iso = 0,    # intrinsic soc (nnn-hoppings)
                    B=Bfield,
                    peierls=bm.peierls_scatter,
                    peierls_lead_L=bm.peierls_lead_L,
                    peierls_lead_R=bm.peierls_lead_R,
                    Lm=0
    )




    n_energy_values = 101
    energy_values = np.linspace(-0.2, 0.2, n_energy_values)
    # tau_inv = calculate_relaxation_time(system, energy_values, syst_params, n_phases=20)
    tau_inv = spin_relaxation_imp(system, energy_values, syst_params, n_phases=20)


    np.savez("../results/spin_relaxation_magnetic_moment_times_hydrogenated_20_phase_range_0,2_both_spin_flips.npz",
            energies=energy_values,
            tau_sz=tau_inv
    )




if __name__ == '__main__':
    main()
