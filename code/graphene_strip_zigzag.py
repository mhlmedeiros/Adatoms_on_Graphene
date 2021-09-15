import kwant
import tinyarray
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product
import builders_with_magnetic_field as bm


FONT_LABELS = 20
font = {'family' : 'serif', 'weight' : 'bold', 'size': FONT_LABELS}
font = {'size': FONT_LABELS}
mpl.rc('font', **font)
plt.rc('text', usetex=True)


def family_colors_H(site):
        if site.family == bm.A:
            color = 'k'
        elif site.family == bm.B:
            color = 'k'
        elif site.family == bm.HA:
            color = 'red'
        elif site.family == bm.HB:
            color = 'red'
        else:
            color = 'red'
        return color

def hopping_colors(site1, site2):
    if site1.family == site2.family:
        color='blue'
    else:
        color='black'
    return color

def hopping_lw(site1, site2):
    return 0.01 if site1.family == site2.family else 0.07

def site_size_function(site):
    if site.family == bm.HA or site.family == bm.HB:
        size = 0.6
    else:
        size = 0.1
    return size


def main():

    adatom_concentration = 0.125  # [% of Carbon atoms]

    ## Define the shape of the system:
    system_width  = 5
    system_length = 10
    shape = bm.Rectangle(width=system_width, length=system_length)

    ## Build the scattering region:
    system = bm.make_graphene_strip(bm.graphene, shape, t=2.6, iso=12.0e-6)

    ## Make the leads:
    leads  = bm.make_graphene_leads(bm.graphene, shape.leads, t=2.6, on_site=0, iso=12.0e-6)

    ## Attach the leads:
    for lead in leads:
        system.attach_lead(lead)

    ## INSERTING ADATOMS:
    adatom_H_params = dict(T = 7.5, eps = 0.16, Lambda_I = -0.21e-3, Lambda_BR = 0.33e-3, Lambda_PIA = -0.77e-3)
    # adatom_F_params = dict(T = 5.5, eps = -2.2, Lambda_I = 3.3e-3, Lambda_BR = 11.2e-3, Lambda_PIA = -7.3e-3)
    system = bm.insert_adatoms_randomly(system, shape, adatom_concentration, adatom_params=adatom_H_params)


    fig, ax = plt.subplots(figsize=(20,5))
    kwant.plot(system,
               site_color=family_colors_H,
               site_size=site_size_function,
               hop_color=hopping_colors,
               hop_lw=hopping_lw,
               site_lw=0.1, ax=ax)
    ax.set_aspect('equal')
    # ax.axis('off')
    plt.show()

    # Calculate the transmission
    Bflux = 0     # in units of quantum of magnetic flux
    Bfield = Bflux / (np.sqrt(3)/2)
    parameters_hand = dict(V=0,
                           t=2.6,
                           B=Bfield,
                           eps_H=0.16,
                           peierls=bm.peierls_scatter,
                           peierls_lead_L=bm.peierls_lead_L,
                           peierls_lead_R=bm.peierls_lead_R,
                           Lm=0,
                        )
    energy_values = np.linspace(-0.5,0.5,300)
    transmission1 = bm.calculate_conductance(system, energy_values, params_dict=parameters_hand)
    #
    # #
    plt.plot(energy_values, transmission1)
    # # plt.plot(energy_values, transmission2)
    plt.show()



if __name__ == '__main__':
    main()
