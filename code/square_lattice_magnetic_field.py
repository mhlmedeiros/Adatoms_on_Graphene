import kwant
import numpy as np
import matplotlib.pyplot as plt
import peierls_substitution as ps


def main():

    Width = 3
    Length = 20
    B_field = 0.5
    L_field = 10


    ##******************************************##
    ##                 "BY HAND"                ##
    ##******************************************##
    syst_sqr = ps.make_system_square_lattice(ps.hopping_by_hand, L=Length, W=Width)
    # kwant.plot(syst_sqr);

    leads_sqr = ps.make_leads_square_lattice(ps.hopping_by_hand, W=Width)
    lead_0_sqr_hand = leads_sqr[0].substituted(peierls='peierls_lead_L')
    lead_1_sqr_hand = leads_sqr[1].substituted(peierls='peierls_lead_R')
    leads_sqr_hand = [lead_0_sqr_hand, lead_1_sqr_hand]

    for lead in leads_sqr:
        syst_sqr.attach_lead(lead)

    # kwant.plot(syst_sqr);
    syst_sqr = syst_sqr.finalized()
    parameters_sqrd = dict(t=1,
                           B=B_field * np.pi,
                           peierls=ps.peierls_syst,
                           peierls_lead_L=ps.peierls_lead_L,
                           peierls_lead_R=ps.peierls_lead_R,
                           Lm=L_field,
                        )

    # ps.plot_conductance(syst_sqr, np.linspace(-10,10,100), parameters_sqrd)

    ##******************************************##
    ##          USING  'kwant.gauge'            ##
    ##******************************************##
    syst_sqr_gauge = ps.make_system_square_lattice(ps.hopping_2, L=Length, W=Width)
    # kwant.plot(syst_sqr_gauge);
    lead_sqr_gauge_0, lead_sqr_gauge_1 = ps.make_leads_square_lattice(ps.hopping_2, W=Width)

    lead_sqr_gauge_0 = lead_sqr_gauge_0.substituted(peierls='peierls_lead_0')
    lead_sqr_gauge_1 = lead_sqr_gauge_1.substituted(peierls='peierls_lead_1')
    leads_sqr_gauge = [lead_sqr_gauge_0, lead_sqr_gauge_1]

    for lead in leads_sqr_gauge:
        syst_sqr_gauge.attach_lead(lead)

    # kwant.plot(syst_sqr_gauge);

    ## FINALIZE THE SYSTEM
    syst_sqr_gauge = syst_sqr_gauge.finalized()

    ## INITIATE THE GAUGE
    gauge_sqr = kwant.physics.magnetic_gauge(syst_sqr_gauge)

    ## DEFINE THE PEIERLS PHASES
    barreira = ps.Barreira(B=B_field, Lm=L_field)
    peierls_sqr_syst, peierls_sqr_lead_0, peierls_sqr_lead_1 = gauge_sqr(barreira, 0, 0)

    ## PUT THE PHASES INTO A DICTIONARY
    parameters_sqrd_gauge = dict(t=1,
                                peierls = peierls_sqr_syst,
                                peierls_lead_0 = peierls_sqr_lead_0,
                                peierls_lead_1 = peierls_sqr_lead_1)

    fig, ax = plt.subplots(figsize=(8,8))
    ps.plot_conductance(syst_sqr, np.linspace(-10,10,202), parameters_sqrd, ax=ax, c='r', label='hand')
    ps.plot_conductance(syst_sqr_gauge, np.linspace(-10,10,202), parameters_sqrd_gauge, ax=ax, c='b', label='kwant')
    ax.legend(fontsize=20)
    plt.show()


if __name__ == '__main__':
    main()
