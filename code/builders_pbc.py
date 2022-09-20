import kwant
import tinyarray
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

#====================================================================#
#                       GLOBAL DEFINITIONS                           #
#====================================================================#
## Lattice definitions:
lat = kwant.lattice.honeycomb(a=1, norbs=2)
A, B = lat.sublattices

zeros_2x2 = tinyarray.array([[0,0],[0,0]])
sigma_0 = tinyarray.array([[1,0],[0,1]])
sigma_x = tinyarray.array([[0,1],[1,0]])
sigma_y = tinyarray.array([[0,-1j],[1j,0]])
sigma_z = tinyarray.array([[1,0],[0,-1]])


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
    def __init__(self, width, length, delta=2, centered=True):
        '''
        Calling the scattering region as strip:
        W = width of the strip
        L = length of the strip
        '''
        self.length = length
        self.delta  = delta
        self.width, self.N = W_to_close(width)

        if centered:
            self.x_inf = -length/2
            self.x_sup = -self.x_inf
            self.y_inf = -width/2
            self.y_sup = -self.y_inf
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

#====================================================================#
#                          System Builders                           #
#====================================================================#

def make_syst_and_leads(shape, V=0, t=1, phi=0):
    syst = make_graphene_strip_PBC(shape, V, t, phi)
    leads = make_graphene_leads_PBC(shape, V, t, phi)
    return syst, leads

def make_graphene_strip_PBC(shape, V, t, phi):
    # DEFINING THE SYSTEM'S SHAPE
    syst = kwant.Builder()
    syst[lat.shape(shape, (0, 0))] = V * sigma_0
    syst[lat.neighbors()] = -t * sigma_0

    sites_x_tags = [s.tag[0] for s in syst.sites() if (s.family == B and s.tag[1]==0)] # can be smaller?
    N = shape.N
    M = max(sites_x_tags) + 1

    for i in range(M):
        syst[A(i-N, 2*N), B(i, 0)] = -t * np.exp(-1j*phi) * sigma_0

    return syst

def make_graphene_leads_PBC(shape, V, t, phi):
    symmetry = kwant.TranslationalSymmetry((-1,0))
    symmetry.add_site_family(lat.sublattices[0], other_vectors=[(-1,2)])
    symmetry.add_site_family(lat.sublattices[1], other_vectors=[(-1,2)])

    lead_0 = kwant.Builder(symmetry)
    lead_0[lat.shape(shape.leads, (0,0))] = V * sigma_0
    lead_0[lat.neighbors()] = -t * sigma_0

    N = shape.N

    lead_0[A(-N, 2*N), B(0, 0)] = -t * np.exp(-1j*phi) * sigma_0
    lead_1 = lead_0.reversed()
    return [lead_0, lead_1]

def W_to_close(W):
    N = max(W // np.sqrt(3), 2)
    return N * np.sqrt(3), int(N)

def assemble_syst(syst, leads):
    for lead in leads:
        syst.attach_lead(lead)
    return syst


def main():

    width = 5
    length = 5

    shape = Rectangle(width, length, centered=False)

    syst, leads = make_syst_and_leads(shape, V=0, t=1, phi=0)

    sites = [s for s in syst.sites()]
    Nsites = len(sites)
    eta = 1/Nsites * 1e6

    print("N. sites = ", Nsites)
    print("Concentration = ", eta)


    kwant.plot(syst)

if __name__ == '__main__':
    main()
