import kwant
import numpy as np
import matplotlib.pyplot as plt

class Barreira:
    def __init__(self, B, Lm):
        self.B = B
        self.Lm = Lm

    def __call__(self, pos):
        B = self.B
        lm = self.Lm/2
        x,_ = pos
        return B * (1/(np.exp(-20*(x + lm))+1) - 1/(np.exp(-20*(x-lm))+1))


##******************************************##
##            BUILDING FUNCTIONS            ##
##******************************************##
def make_circ_system(r=10, w=2.0, pot=0.1):

    #### Define the scattering region. ####
    # circular scattering region
    def circle(pos):
        x, y = pos
        return x ** 2 + y ** 2 < r ** 2

    syst = kwant.Builder()

    # w: width and pot: potential maximum of the p-n junction
    def potential(site):
        (x, y) = site.pos
        d = y * cos_30 + x * sin_30
        return pot * tanh(d / w)

    syst[graphene.shape(circle, (0, 0))] = potential

    # specify the hoppings of the graphene lattice in the
    # format expected by builder.HoppingKind
    hoppings = (((0, 0), a, b), ((0, 1), a, b), ((-1, 1), a, b))
    syst[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -1

    # Modify the scattering region
    del syst[a(0, 0)]
    syst[a(-2, 1), b(2, 2)] = -1

    #### Define the leads. ####
    # left lead
    sym0 = kwant.TranslationalSymmetry(graphene.vec((-1, 0)))

    def lead0_shape(pos):
        x, y = pos
        return (-0.4 * r < y < 0.4 * r)

    lead0 = kwant.Builder(sym0)
    lead0[graphene.shape(lead0_shape, (0, 0))] = -pot
    lead0[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -1

    # The second lead, going to the top right
    sym1 = kwant.TranslationalSymmetry(graphene.vec((0, 1)))

    def lead1_shape(pos):
        v = pos[1] * sin_30 - pos[0] * cos_30
        return (-0.4 * r < v < 0.4 * r)

    lead1 = kwant.Builder(sym1)
    lead1[graphene.shape(lead1_shape, (0, 0))] = pot
    lead1[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -1

    return syst, [lead0, lead1]

def make_rect_system(W=10, L=20, pot=0.1, t=1):

    #### Define the scattering region. ####
    def rectangle(pos):
        x, y = pos
        return -W/2 < y < W/2 and -L/2 < x < L/2

    syst = kwant.Builder()
    syst[graphene.shape(rectangle, (0, 0))] = -pot

    # specify the hoppings of the graphene lattice in the
    # format expected by builder.HoppingKind
    hoppings = (((0, 0), a, b), ((0, 1), a, b), ((-1, 1), a, b))
    syst[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -t

    #### Define the leads. ####
    # left lead
    sym0 = kwant.TranslationalSymmetry(graphene.vec((-1, 0)))

    def lead0_shape(pos):
        x, y = pos
        return (-W/2< y < W/2)

    lead0 = kwant.Builder(sym0)
    lead0[graphene.shape(lead0_shape, (0, 0))] = -pot
    lead0[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -t

    # The second lead, going to the right
    lead1 = lead0.reversed()
    return syst, [lead0, lead1]

def make_whole_sqr_syst_by_hand(W, L, V=0, initialpoint=(0,0)):
    #************************************#
    #    Define the scattering region.   #
    #************************************#
    def rectangle(pos):
        x, y = pos
        return -W/2 < y < W/2 and -L/2 <= x <= L/2

    lat = kwant.lattice.square()
    syst = kwant.Builder()

    syst[lat.shape(rectangle, initialpoint)] = V # on-site term
    syst[lat.neighbors()] = hopping_func # hopping term

def make_system_square_lattice(hopping_func, W=4.5, L=10, V=0, initialpoint=(0,0)):
    #************************************#
    #    Define the scattering region.   #
    #************************************#
    def rectangle(pos):
        x, y = pos
        return -W/2 < y < W/2 and -L/2 <= x <= L/2
    lat = kwant.lattice.square()
    syst = kwant.Builder()
    syst[lat.shape(rectangle, initialpoint)] = V # on-site term
    syst[lat.neighbors()] = hopping_func # hopping term
    return syst

def make_leads_square_lattice(hopping_func, W=4.5, V=0):
    def lead_shape(pos):
        y = pos[1]
        return -W/2 < y < W/2

    lat = kwant.lattice.square()

    # LEFT LEAD:
    symmetry_0 = kwant.TranslationalSymmetry((-1,0))
    lead_0 = kwant.Builder(symmetry_0)
    lead_0[lat.shape(lead_shape, (0,0))] = V
    lead_0[lat.neighbors()] = hopping_func

    # RIGHT LEAD:
    lead_1 = lead_0.reversed()

    return [lead_0, lead_1]


##******************************************##
##                 "BY HAND"                ##
##******************************************##
def hopping_by_hand(Site1, Site2, t, B, Lm, peierls):
    return -t * peierls(Site1, Site2, B, Lm)

def peierls_syst(Site1, Site2, B, Lm):
    (x_i, y_i) = Site1.pos # Target
    (x_j, y_j) = Site2.pos # Source
    x_i, x_j = change_x(x_i, Lm), change_x(x_j, Lm)
    theta = B/2*(x_i + x_j)*(y_i - y_j)
    return np.exp(1j*theta)

def peierls_lead_L(Site1, Site2, B, Lm):
    (x_i, y_i) = Site1.pos # Target
    (x_j, y_j) = Site2.pos # Source
    theta = -B/2 * Lm * (y_i - y_j)
    return np.exp(1j*theta)

def peierls_lead_R(Site1, Site2, B, Lm):
    (x_i, y_i) = Site1.pos # Target
    (x_j, y_j) = Site2.pos # Source
    theta = B/2 * Lm * (y_i - y_j)
    return np.exp(1j*theta)

def change_x(x, Lm):
    if (-Lm/2) <= x <= (Lm/2): return x
    elif x > (Lm/2): return Lm/2
    else: return -Lm/2

def hopping_2(a, b, t, peierls):
    return -t * peierls(a, b)


##******************************************##
##          USING  'kwant.gauge'            ##
##******************************************##
sin_30, cos_30 = (1/2, np.sqrt(3)/2)
graphene = kwant.lattice.general([(1, 0), (sin_30, cos_30)],
                                 [(0, 0), (0, 1 / np.sqrt(3))],name='Graphene')
a, b = graphene.sublattices

def make_graphene_peierls(hopping_func, W=4.5, L=20, V=0):

    #### Define the scattering region. ####
    # circular scattering region
    def rectangle(pos):
        x, y = pos
        return -W/2 < y < W/2 and -L/2 <= x <= L/2

    syst = kwant.Builder()
    syst[graphene.shape(rectangle, (0, 0))] = V

    # specify the hoppings of the graphene lattice in the
    # format expected by builder.HoppingKind
    hoppings = (((0, 0), a, b), ((0, 1), a, b), ((-1, 1), a, b))
    syst[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = hopping_func

    return syst

def make_graphene_leads_peierls(hopping_func, W=4.5, V=0):
    def lead_shape(pos):
        y = pos[1]
        return -W/2 < y < W/2

    symmetry = kwant.TranslationalSymmetry((-1,0))
    lead_0 = kwant.Builder(symmetry)
    lead_0[graphene.shape(lead_shape, (0,0))] = V
    a, b = graphene.sublattices
    hoppings = (((0, 0), a, b), ((0, 1), a, b), ((-1, 1), a, b))
    lead_0[(kwant.builder.HoppingKind(*hopping) for hopping in hoppings)] = hopping_func
    lead_1 = lead_0.reversed()
    return [lead_0, lead_1]


##******************************************##
##              VISUALIZATION               ##
##******************************************##
def family_colors(site):
    return 'w' if site.family == a else 'k' if site.family == b else 'r'

def plot_conductance(syst, energies, params_dict, ax=None, **kargs):
    # Compute transmission as a function of energy
    data = []
    for energy in energies:
        smatrix = kwant.smatrix(syst, energy, params=params_dict)
        data.append(smatrix.transmission(0, 1))

    if ax == None:
        fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(energies, data, **kargs)
    ax.set_xlabel("energy [t]")
    ax.set_ylabel("conductance [e^2/h]")
