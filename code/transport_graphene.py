
import kwant
import tinyarray
import numpy as np
import matplotlib.pyplot as plt

import graphene_builders as gb

def main():
    ## Define the shape of the system
    shape = gb.Rectangle(W=20, L=5)
    graphene = gb.graphene # making an alias
    A, B = graphene.sublattices # splitting into sublattices


    ## Make the scattering region
    zeros, s_0 = gb.zeros_2x2, gb.sigma_0
    t_hop = 2.6      # eV
    lambda_I = 12e-6 # eV [12 mu eV]
    system = gb.make_graphene_strip(graphene, shape, on_site=zeros, t=t_hop, iso=lambda_I)

    ## Make the leads
    leads  = gb.make_graphene_leads(graphene, shape.leads, on_site=zeros,  t=t_hop, iso=lambda_I)

    ## Attach the leads
    for lead in leads:
        system.attach_lead(lead)

    # Calculate the transmission
    energy_values = np.linspace(-2,2,100)
    transmission1 = gb.calculate_conductance(system, energy_values)

    ## Insert (or not) the adatom and its hoppings
    pos_tag = (0,0)  # (-5,10)
    sub_lat = A      # B
    adatom_params = dict(T = 7.5, eps_h = 0.16, L_I = -0.21e-3,
                            L_BR = 0.33e-3, L_PIA = -0.77e-3)
                            
    gb.insert_adatom(system, pos_tag, sub_lat,  **adatom_params)

    # Verify the system shape (plot)
    # gb.plot_system(system)

    ## Calculate the band structure from left lead
    # gb.plot_bands(leads[0])

    # Calculate the transmission
    energy_values = np.linspace(-2,2,100)
    transmission2 = gb.calculate_conductance(system, energy_values)

    plt.plot(energy_values, transmission1)
    plt.plot(energy_values, transmission2)
    plt.show()



if __name__ == '__main__':
    main()
