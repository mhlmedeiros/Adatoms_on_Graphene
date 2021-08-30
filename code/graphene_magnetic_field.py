import kwant
import numpy as np
import matplotlib.pyplot as plt
import peierls_substitution as ps


def plot_syst(syst):
    fig, ax = plt.subplots(figsize=(15,5))
    ax.set_aspect('equal')
    kwant.plot(syst, site_color=ps.family_colors,
                     site_lw=0.1, colorbar=False, ax=ax);

def main():
    Width = 4.5
    Length = 20
    B_field = 2.0
    L_field = 10


    ##******************************************##
    ##                 "BY HAND"                ##
    ##******************************************##
    system_hand = ps.make_graphene_peierls(ps.hopping_by_hand, W=Width, L=Length)
    lead_0_hand, lead_1_hand = ps.make_graphene_leads_peierls(ps.hopping_by_hand, W=Width)

    lead_0_hand = lead_0_hand.substituted(peierls='peierls_lead_L')
    lead_1_hand = lead_1_hand.substituted(peierls='peierls_lead_R')
    leads_hand = [lead_0_hand, lead_1_hand]

    # Attach the leads to the system.
    for lead in leads_hand:
        system_hand.attach_lead(lead)

    # plot_syst(system_hand)

    system_hand = system_hand.finalized()

    parameters_hand = dict(t=1,
                       B=B_field*np.pi,
                       peierls= ps.peierls_syst,
                       peierls_lead_L= ps.peierls_lead_L,
                       peierls_lead_R= ps.peierls_lead_R,
                       Lm=L_field,
                    )
    # ps.plot_conductance(system_hand, np.linspace(-4,4,202), parameters_hand)

    ##******************************************##
    ##          USING  'kwant.gauge'            ##
    ##******************************************##
    system_gauge = ps.make_graphene_peierls(ps.hopping_2, W=Width, L=Length)
    lead_0_gauge, lead_1_gauge = ps.make_graphene_leads_peierls(ps.hopping_2, W=Width)

    lead_0_gauge = lead_0_gauge.substituted(peierls='peierls_lead_0')
    lead_1_gauge = lead_1_gauge.substituted(peierls='peierls_lead_1')
    leads_gauge = [lead_0_gauge, lead_1_gauge]

    for lead in leads_gauge:
        system_gauge.attach_lead(lead)

    system_gauge = system_gauge.finalized()
    gauge = kwant.physics.magnetic_gauge(system_gauge)

    barreira = ps.Barreira(B=B_field, Lm=L_field)

# x_plot_test = np.linspace(-10,10,201)
# y_plot_test = [barreira((x,1)) for x in x_plot_test]
# plt.plot(x_plot_test, y_plot_test)

    peierls_syst, peierls_lead_0, peierls_lead_1 = gauge(barreira, 0, 0)

    parameters_gauge = dict(t=1,
                    peierls=peierls_syst,
                    peierls_lead_0=peierls_lead_0,
                    peierls_lead_1=peierls_lead_1)

    fig, ax = plt.subplots(figsize=(8,8))
    ps.plot_conductance(system_hand, np.linspace(-4,4,202), parameters_hand, ax)
    ps.plot_conductance(system_gauge, np.linspace(-4,4,202), parameters_gauge, ax)
    plt.show()


if __name__ == '__main__':
    main()
