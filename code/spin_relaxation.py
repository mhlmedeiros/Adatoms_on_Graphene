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


FONT_LABELS = 20
font = {'family' : 'serif', 'weight' : 'bold', 'size': FONT_LABELS}
font = {'size': FONT_LABELS}
mpl.rc('font', **font)
plt.rc('text', usetex=True)

HBAR  = sci.physical_constants['reduced Planck constant in eV s'][0]



def spin_flip_probability(smatrix):
    """
    This function returns the spin-flip probability per mode for
        * in-plane spins: gamma_sx and gamma_sy
        * out-of-plane spins: gamma_sz
    and the number of propagating modes:
        * modes_per_spin

    When modes_per_spin == 0, the function returns four zeros.

    """
    modes_per_spin = smatrix.num_propagating(0)/2

    if modes_per_spin:
        t_sz_list, r_sz_list = calculate_sz_matrices(smatrix)
        t_sx_list, r_sx_list = sz_to_sx_matrices(t_sz_list, r_sz_list)
        t_sy_list, r_sy_list = sz_to_sy_matrices(t_sz_list, r_sz_list)
        gamma_sz = (la.norm(t_sz_list[1])**2 + la.norm(t_sz_list[2])**2
                 + la.norm(r_sz_list[1])**2 + la.norm(r_sz_list[2])**2)/modes_per_spin
        gamma_sx = (la.norm(t_sx_list[1])**2 + la.norm(t_sx_list[2])**2
                 + la.norm(r_sx_list[1])**2 + la.norm(r_sx_list[2])**2)/modes_per_spin
        gamma_sy = (la.norm(t_sy_list[1])**2 + la.norm(t_sy_list[2])**2
                 + la.norm(r_sy_list[1])**2 + la.norm(r_sy_list[2])**2)/modes_per_spin
    else:
        gamma_sx = 0
        gamma_sy = 0
        gamma_sz = 0
    return gamma_sx, gamma_sy, gamma_sz, modes_per_spin


def calculate_sz_matrices(smatrix):
    """
    This function returns two 2x2 matrices that contain the
    reflection and transmission resolved by spin projections:

    r_matrix = [[R_up_up, R_up_dn],
                [R_dn_up, R_dn_dn]]

    t_matrix = [[T_up_up, T_up_dn],
                [T_dn_up, T_dn_dn]]

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


def sz_to_sx_matrices(t_list, r_list):
    return change_to_sx(t_list), change_to_sx(r_list)


def sz_to_sy_matrices(t_list, r_list):
    return change_to_sy(t_list), change_to_sy(r_list)


def change_to_sx(matrices):
    uu, ud, du, dd = matrices
    new_uu =  0.5 * (uu + ud + du + dd)
    new_ud =  0.5 * (uu - ud + du - dd)
    new_du =  0.5 * (uu + ud - du - dd)
    new_dd =  0.5 * (uu - ud - du + dd)
    return [new_uu, new_ud, new_du, new_dd]


def change_to_sy(matrices):
    # TODO: CORRIGIR TRANSFORMATION (COMPLEX CONJUGATION)
    uu, ud, du, dd = matrices
    new_uu =  0.5 * (uu + 1j*ud - 1j*du + dd)
    new_ud =  0.5 * (uu - 1j*ud - 1j*du - dd)
    new_du =  0.5 * (uu + 1j*ud + 1j*du - dd)
    new_dd =  0.5 * (uu - 1j*ud + 1j*du + dd)
    return [new_uu, new_ud, new_du, new_dd]


def transmission_matrices(smatrix, lead_source, lead_target):
    # modes_per_spin = smatrix.num_propagating(lead_source)/2
    # print("# modes = ", modes_per_spin)
    modes_per_spin = 1
    # print("N-modes = ", modes_per_spin)
    T_uu = smatrix.submatrix((lead_target, 0), (lead_source, 0)) * 1/modes_per_spin
    T_ud = smatrix.submatrix((lead_target, 0), (lead_source, 1)) * 1/modes_per_spin
    T_du = smatrix.submatrix((lead_target, 1), (lead_source, 0)) * 1/modes_per_spin
    T_dd = smatrix.submatrix((lead_target, 1), (lead_source, 1)) * 1/modes_per_spin
    return T_uu, T_ud, T_du, T_dd


def show_matrices(t_list, r_list, modes_per_spin, spin='z'):
    T_prob = [la.norm(T)**2/modes_per_spin for T in t_list]
    R_prob = [la.norm(R)**2/modes_per_spin for R in r_list]
    print("\nTRANSMISSION S{}: ".format(spin))
    print("[[{:.3f}, {:.3f}],\n [{:.3f}, {:.3f}]]".format(*T_prob))
    print("\nREFLECTION S{}: ".format(spin))
    print("[[{:.3f}, {:.3f}],\n [{:.3f}, {:.3f}]]".format(*R_prob))


def test_matrices(system, energy, params):
    smatrix = kwant.smatrix(system.finalized(), energy, params=params)
    modes_per_spin = smatrix.num_propagating(0)/2

    t_sz_list, r_sz_list = calculate_sz_matrices(smatrix)
    t_sx_list, r_sx_list = sz_to_sx_matrices(t_sz_list, r_sz_list)
    t_sy_list, r_sy_list = sz_to_sy_matrices(t_sz_list, r_sz_list)

    show_matrices(t_sz_list, r_sz_list, modes_per_spin, spin='z')
    show_matrices(t_sx_list, r_sx_list, modes_per_spin, spin='x')
    show_matrices(t_sy_list, r_sy_list, modes_per_spin, spin='y')



def main():
    ## DEFINE THE STRIP
    system_width  = 130
    system_length = 62
    shape = bm.Rectangle(width=system_width, length=system_length, pbc=True)

    # Build the scattering region:
    system = bm.make_graphene_strip(bm.graphene, shape)

    # Make the leads:
    leads  = bm.make_graphene_leads(bm.graphene, shape)


    # Attach the leads:
    for lead in leads:
        system.attach_lead(lead)
    #
    ## INSERT THE ADATOM:
    number_of_carbons =  len(system.sites())
    eta = 1/number_of_carbons
    density_ppm = eta * 1e6 # [ppm]
    density_percent = density_ppm * 1e-4

    L_I   = -0.21e-3 ## INTRINSIC SOC
    L_BR  =  0.33e-3 ## RASHBA SOC
    L_PIA = -0.77e-3 ## PSEUDO INVERSION ASYMMETRY

    H_params = dict(T = 7.5, eps = 0.16, Lambda_I = L_I, Lambda_BR = L_BR, Lambda_PIA = L_PIA)
    H_params_only_ISO = dict(T = 7.5, eps = 0.16, Lambda_I = L_I, Lambda_BR =    0, Lambda_PIA =     0)
    H_params_only_BR  = dict(T = 7.5, eps = 0.16, Lambda_I =   0, Lambda_BR = L_BR, Lambda_PIA =     0)
    H_params_only_PIA = dict(T = 7.5, eps = 0.16, Lambda_I =   0, Lambda_BR =    0, Lambda_PIA = L_PIA)
    system = bm.insert_adatoms_randomly(system, shape, density_percent, H_params_only_PIA)


    ## CALCULATE THE TRANSMISSION COEFFICIENTS
    Bfield = 0
    syst_params = dict(V=0,   # on-site C-atoms
                    t=2.6, # hoppings C-atoms
                    phi=0, # PBC hopping phase
                    lambda_iso = 0, # intrinsic soc (nnn-hoppings)
                    B=Bfield,
                    peierls=bm.peierls_scatter,
                    peierls_lead_L=bm.peierls_lead_L,
                    peierls_lead_R=bm.peierls_lead_R,
                    Lm=0)

    # test_matrices(system, 0.4, syst_params)

    n_energy_values = 101 # MAYBE MORE ENERGY VALUES (CLUSTER?)
    energy_values = np.linspace(-0.4, 0.4, n_energy_values)
    tau_sx = np.empty_like(energy_values)
    tau_sy = np.empty_like(energy_values)
    tau_sz = np.empty_like(energy_values)

    n_phases = 20 ## MAYBE MORE PHASE VALUES (CLUSTER?)
    phi_values = np.linspace(0, 2*np.pi, n_phases)

    # energy_test = 0.4
    # smatrix = kwant.smatrix(system.finalized(), energy_test, params=syst_params)
    # g_sx, g_sy, g_sz = spin_flip_probability(smatrix)

    CONST = 4 * syst_params['t']/HBAR * eta * shape.width

    for ind in range(n_energy_values):
        print("Calculating the smatrix for E = {:2.2f} ... ".format(energy_values[ind]), end='')
        gamma_sx = 0
        gamma_sy = 0
        gamma_sz = 0
        n_phases_with_propag = 0

        for phi in phi_values:
            syst_params["phi"] = phi
            smatrix = kwant.smatrix(system.finalized(), energy_values[ind], params=syst_params)
            g_sx, g_sy, g_sz, modes_per_spin = spin_flip_probability(smatrix)
            gamma_sx += g_sx
            gamma_sy += g_sy
            gamma_sz += g_sz

            if modes_per_spin:
                n_phases_with_propag += 1

        n_phases_with_propag = max(n_phases_with_propag, 1)

        tau_sx[ind] = CONST * gamma_sx/n_phases_with_propag
        tau_sy[ind] = CONST * gamma_sy/n_phases_with_propag
        tau_sz[ind] = CONST * gamma_sz/n_phases_with_propag
        print("OK")

    # TODO: RUN FOR ONLY-PIA:
    np.savez("../results/spin_relaxation_times_hydrogenated_20_phases_test_only_PIA.npz",
                                                            energies=energy_values,
                                                            tau_sx=tau_sx,
                                                            tau_sy=tau_sy,
                                                            tau_sz=tau_sz,
    )




if __name__ == '__main__':
    main()
