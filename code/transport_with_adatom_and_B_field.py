"""
DOC-STRING:
"""
import kwant
import numpy as np
import matplotlib.pyplot as plt

import builders_with_magnetic_field as bm

NN_SPACING = 0.142e-9                    ## Carbon atoms separation [m]
A_BRAVAIS = np.sqrt(3) * NN_SPACING        ## Bravais lattice spacing
PHI_0 = 2.067833848e-15                    ## [Wb] = [Tm^2]
CELL_AREA = np.sqrt(3)/2                   ## unit cell area considering a == 1
UNIT_CELL_AREA = CELL_AREA * A_BRAVAIS**2  ## m^2


def tesla_to_au(Bfield):
    """
    Convert the field unit from Tesla to PHI_0/CELL_AREA
    As the code considers A_BRAVAIS == 1 and PHI_0 == 1
    this conversion is necessary to pass the correct value
    of magnetic field to the solvers.
    """
    Bflux = field_to_flux(Bfield)  # flux of magnetic field in units of quantum of flux
    return Bflux/CELL_AREA


def field_to_flux(field):
    """
    Calculate the magnetic field flux
    in units of PHI_0.
    """
    return field * UNIT_CELL_AREA/PHI_0


def to_tesla(Bfield):
    """
    Convert the field unit from Tesla to PHI_0/CELL_AREA
    As the code considers A_BRAVAIS == 1 and PHI_0 == 1
    this conversion is necessary to pass the correct value
    of magnetic field to the solvers.
    """
    Bflux = Bfield * CELL_AREA
    return flux_to_field(Bflux)


def flux_to_field(flux):
    """
    Given a flux in units of PHI_0, returns
    the field in Tesla.
    """
    return flux * PHI_0/UNIT_CELL_AREA


def define_system(system_width, system_length, n_adatoms, H_params):
    """
    Build a finite graphene strip with a given number of non-magnetic adatoms:
    :param system_width: Float
    :param system_length: Float
    :param n_adatoms: Int
    :param H_params: Dict
    :return system: Builder
    """
    ## DEFINE THE STRIP
    shape = bm.Rectangle(width=system_width, length=system_length, centered=False, pbc=False)

    # Build the scattering region:
    system = bm.make_graphene_strip(bm.graphene, shape)

    # Make the leads:
    leads  = bm.make_graphene_leads(bm.graphene, shape)

    # Attach the leads:
    for lead in leads:
        system.attach_lead(lead)

    ## INSERT THE ADATOM:
    number_of_carbons =  len(system.sites())   # count the number of sites
    eta = n_adatoms/number_of_carbons  # define the concentration of adatoms
    density_percent = eta * 100                # concentration in percentage
    system = bm.insert_adatoms_randomly(system, shape, density_percent, H_params, verbatim=False)
    print(f'Concentration of adatoms = {round(density_percent,3)}%')
    return system


def calculate_transmission_by_energy(syst, energy_values, syst_params, per_mode=False):
    """
    Calculate the transmission for each given value of energy:
    :param syst: Builder
    :param energy_values: Array
    :param syst_params: Dict
    :param per_mode: Bool
    :return: Array
    """
    return np.array([transmission_single_energy(syst, E, syst_params, per_mode) for E in energy_values])


def transmission_single_energy(syst, energy, syst_params, per_mode=False):
    """
    Calculate the transmission for a single value of energy:
    :param syst: Builder
    :param energy: Float
    :param syst_params: Dict
    :param per_mode: Bool
    :return: Float
    """
    smatrix = kwant.smatrix(syst.finalized(), energy, params=syst_params)
    n_modes = smatrix.num_propagating(0) if per_mode else 0
    return smatrix.transmission(1, 0)/max(1, n_modes)


def transmission_comparison_single(system_width, system_length, n_adatoms, H_params, syst_params, energy, N_config):
    transmission_B_zero = np.zeros(N_config)
    transmission_B = np.zeros(N_config)
    B_flux = syst_params['B']
    Bfield_T = to_tesla(B_flux)

    for i in range(N_config):
        print(f"Calculating {i+1}/{N_config}", end=' ')
        syst_params['B'] = B_flux
        print(f"with B = {to_tesla(B_flux)} and 0 Tesla.")
        system = define_system(system_width, system_length, n_adatoms, H_params)
        transmission_B[i] = transmission_single_energy(system, energy, syst_params, per_mode=True)
        syst_params['B'] = 0
        transmission_B_zero[i] = transmission_single_energy(system, energy, syst_params, per_mode=True)

    path = "../results/conduction/"
    name_part_1 = "Energy_{:.3f}_eV_{:d}_adatoms_0_and_{:g}_Tesla".format(energy, n_adatoms, Bfield_T)
    name_part_2 = f"_{N_config}_nconfig_width_{system_width}_length_{system_length}"
    name_part_3 = f"_zigzag.npz"
    full_name = path + name_part_1 + name_part_2 + name_part_3
    print("Saving results in : ", full_name)
    np.savez(full_name, transmission_B=transmission_B, transmission_B_zero=transmission_B_zero)


def transmission_comparison_array(system_width, system_length, n_adatoms, H_params, syst_params, energy_values):
    system = define_system(system_width, system_length, n_adatoms, H_params)
    transmission_B = calculate_transmission_by_energy(system, energy_values, syst_params)
    Bflux = syst_params['B'] * np.sqrt(3)/2
    syst_params['B'] = 0
    transmission_B_zero = calculate_transmission_by_energy(system, energy_values, syst_params)

    path = "../results/conduction/"
    name_part_1 = f"with_{n_adatoms}_adatoms_Bflux_0_and_{Bflux}"
    name_part_2 = f"_width_{system_width}_length_{system_length}"
    name_part_3 = f"_N_energies_{len(energy_values)}_zigzag.npz"
    full_name = path + name_part_1 + name_part_2 + name_part_3

    print("Saving results in : ", full_name)
    np.savez(full_name, energy_values=energy_values,
                        transmission_B=transmission_B,
                        transmission_B_zero=transmission_B_zero)



L_I   = -0.21e-3 ## INTRINSIC SOC
L_BR  =  0.33e-3 ## RASHBA SOC
L_PIA = -0.77e-3 ## PSEUDO INVERSION ASYMMETRY
H_params = dict(T = 7.5, eps = 0.16, Lambda_I = L_I, Lambda_BR = L_BR, Lambda_PIA = L_PIA, exchange = 0)
H_params_without_SO = dict(T = 7.5, eps = 0.16, Lambda_I = 0, Lambda_BR =    0, Lambda_PIA =     0, exchange = 0)
H_params_only_ISO = dict(T = 7.5, eps = 0.16, Lambda_I = L_I, Lambda_BR =    0, Lambda_PIA =     0, exchange = 0)
H_params_only_BR  = dict(T = 7.5, eps = 0.16, Lambda_I =   0, Lambda_BR = L_BR, Lambda_PIA =     0, exchange = 0)
H_params_only_PIA = dict(T = 7.5, eps = 0.16, Lambda_I =   0, Lambda_BR =    0, Lambda_PIA = L_PIA, exchange = 0)


syst_params =  dict(V = 0,
                    t = 2.6,
                    phi = 0,
                    lambda_iso = 12e-6,
                    B = 0,
                    x_Binf = 0,
                    x_Bsup = 0,
                    peierls = bm.peierls_scatter,
                    peierls_pbc = bm.peierls_pbc,
                    peierls_lead_L = bm.peierls_lead_L,
                    peierls_pbc_L = bm.peierls_pbc_L,
                    peierls_lead_R = bm.peierls_lead_R,
                    peierls_pbc_R = bm.peierls_pbc_R)


def main():
    ## DEFINE THE STRIP
    system_width  = 40
    system_length = 30
    number_of_adatoms = 4
    adatoms_params = H_params_without_SO

    # kwant.plot(system)

    ## MAGNETIC FIELD DEFINITION:
    Bfield_tesla = 1.0 # magnetic field in Tesla

    syst_params['x_Binf'] = 0
    syst_params['x_Bsup'] = system_length
    syst_params['B'] = tesla_to_au(Bfield_tesla)


    ############################################################################
    #                 The Actual Calculation happens down here                 #
    ############################################################################
    # energy_values = np.linspace(-0.25, 0.25, 50)
    # transmission_comparison_array(system_width, system_length, number_of_adatoms,
    #                               adatoms_params, syst_params, energy_values)

    energy = 0.01
    N_config = 200
    transmission_comparison_single(system_width, system_length, number_of_adatoms,
                                adatoms_params, syst_params, energy, N_config)

    # plt.plot(transmission_B_zero)
    # plt.plot(transmission_B)
    # plt.show()

    ## SAVING COMPARISON
    # path = "../results/conduction/"
    # name_part_1 = f"Energy_{energy}_with_{number_of_adatoms}_adatoms_Bflux_0_and_{Bflux}"
    # name_part_2 = f"_nconfig_{N_config}_width_{system_width}_length_{system_length}"
    # name_part_3 = f"_zigzag.npz"
    # full_name = path + name_part_1 + name_part_2 + name_part_3
    # print("Saving results in : ", full_name)
    # np.savez(full_name, transmission_B=transmission_B, transmission_B_zero=transmission_B_zero)
    #
    #
    # energy_values = np.linspace(-0.25, 0.25, 50)
    # syst_params['B'] = 0
    # transmission_B_zero = calculate_transmission_by_energy(system, energy_values, syst_params)
    # syst_params['B'] = B_field
    # transmission_B = calculate_transmission_by_energy(system, energy_values, syst_params)



    # plt.plot(energy_values, transmission)
    # plt.show()
    # transmission_single_energy(system, 0.01, syst_params, per_mode=True)


if __name__ == '__main__':
    main()
