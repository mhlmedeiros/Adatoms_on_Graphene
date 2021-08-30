import kwant
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
#import peierls_substitution as ps


class Rectangle:
    def __init__(self, width, length):
        self.width = width
        self.length = length
    def __call__(self, pos):
        x, y = pos
        return 0 <= y <= self.width and 0 <= x <= self.length   

class Square(Rectangle):
    def __init__(self, side):
        self.width = side
        self.length = side

class LineVertical:
    def __init__(self, width):
        self.width = width
    def __call__(self, pos):
        x, y = pos
        return 0 <= y <= self.width

class LineHorizontal:
    def __init__(self, length):
        self.length = length
    def __call__(self, pos):
        x, y = pos
        return 0 <= x <= self.length


def hopping(site1, site2, t, B, peierls):
    return -t * peierls(site1, site2, B)


def onsite(site, V):
    x,y = site.pos
    return V


def peierls(site1, site2, B):
    xi, yi = site1.pos # target
    xj, yj = site2.pos # source
    area = np.sqrt(3)/2
    theta = - B/area * (xi-xj) * (yi+yj)/2
    return np.exp(2j*np.pi*theta)


def peierls_vertical(site1, site2, B):
    xi, yi = site1.pos # target
    xj, yj = site2.pos # source
    area = np.sqrt(3)/2
    theta = B/area * (xi+xj) * (yi-yj)/2
    return np.exp(2j*np.pi*theta)


def make_system(shape_function):
    lattice = kwant.lattice.honeycomb()
    a, b = lattice.sublattices
    system = kwant.Builder()
    system[lattice.shape(shape_function, (0,0))] = onsite
    system[lattice.neighbors()] = hopping
    system.eradicate_dangling()
    return system

def make_lead(shape_function):
    lattice = kwant.lattice.honeycomb()
    a, b = lattice.sublattices
    
    symmetry = kwant.TranslationalSymmetry((-1,0))
    symmetry.add_site_family(lattice.sublattices[0], other_vectors=[(-1,2)])
    symmetry.add_site_family(lattice.sublattices[1], other_vectors=[(-1,2)])
    
    lead_L = kwant.Builder(symmetry)
    lead_L[lattice.shape(shape_function, (0,0))] = onsite
    lead_L[lattice.neighbors()] = hopping
    lead_L.eradicate_dangling()
    
    return lead_L


def make_armchair_lead(shape_function):
    lattice = kwant.lattice.honeycomb()
    a, b = lattice.sublattices
    
    symmetry = kwant.TranslationalSymmetry((0,np.sqrt(3)))
    #symmetry.add_site_family(lattice.sublattices[0], other_vectors=[(-1,2)])
    #symmetry.add_site_family(lattice.sublattices[1], other_vectors=[(-1,2)])
    
    lead_D = kwant.Builder(symmetry)
    lead_D[lattice.shape(shape_function, (0,0))] = onsite
    lead_D[lattice.neighbors()] = hopping
    lead_D.eradicate_dangling()

    return lead_D



if __name__ == "__main__":
    side = 60
    #square = Square(side)
    #syst = make_system(shape_function=square)
    #lead_shape = LineVertical(side)
    lead_shape = LineHorizontal(side)
    #lead = make_lead(lead_shape)
    lead = make_armchair_lead(lead_shape)
    
    #kwant.plot(syst)
    kwant.plot(lead)

    Bflux_values = np.linspace(0, 1, 1001)
    levels = []
    for Bflux in Bflux_values:
        parameters = dict(t=1, B=Bflux, V=0, peierls=peierls_vertical)
        #matrix = syst.finalized().hamiltonian_submatrix(params=parameters)
        #matrix = left_lead.finalized().hamiltonian_submatrix(params=parameters)
        #values, vectors = la.eigh(matrix)
        #levels.append(values)
        bands = kwant.physics.Bands(lead.finalized(), params=parameters)
        levels.append(bands(0))

    fig, ax = plt.subplots(figsize=(5,5))
    #ax.plot(Bflux_values, levels, c="k", lw=0.5)
    ax.plot(Bflux_values, levels, ',', c="k")
    ax.set_xlim(0,1)
    ax.set_ylim(-3,3)
    ax.tick_params(labelsize=15)
    ax.set_xlabel(r'$\Phi/\Phi_0$', fontsize=20)
    ax.set_ylabel(r'$\varepsilon$', fontsize=20)
    plt.tight_layout()
    plt.show()


