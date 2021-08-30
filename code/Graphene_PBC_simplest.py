
import kwant
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

## GLOBAL DEFINITIONS:
lat = kwant.lattice.honeycomb(a=1, norbs=1)
A, B = lat.sublattices

class Rectangle:
    def __init__(self, W, L):
        self.W = W
        self.L = L
    def __call__(self, pos):
        """
        Define the scattering region's shape.
        """
        W, L = self.W, self.L
        x, y = pos
        return 0 < y <= W and 0 <= x < L

class LeadShape:
    def __init__(self, W):
        self.W = W
    def __call__(self, pos):
        y = pos[1]
        return 0 < y <= self.W

def make_syst_and_leads(W=4.5, L=20, V=0, t=1, phi=0):
    syst = make_graphene_strip_PBC(W, L, V, t, phi)
    leads = make_graphene_leads_PBC(W, V, t, phi)
    return syst, leads

def make_graphene_strip_PBC(W, L, V, t, phi):
    # DEFINING THE SYSTEM'S SHAPE
    W_new, N = W_to_close(W)
    rectangle = Rectangle(W_new, L)

    syst = kwant.Builder()
    syst[lat.shape(rectangle, (0, 0))] = V
    syst[lat.neighbors()] = -t

    sites_x_tags = [s.tag[0] for s in syst.sites() if s.family == B]
    M = max(sites_x_tags) + 1

    for i in range(M):
        syst[A(i-N, 2*N), B(i, 0)] = -t * np.exp(-1j*phi)

    return syst

def make_graphene_leads_PBC(W, V, t, phi):
    W_new, N = W_to_close(W)
    lead_shape = LeadShape(W_new)

    symmetry = kwant.TranslationalSymmetry((-1,0))
    symmetry.add_site_family(lat.sublattices[0], other_vectors=[(-1,2)])
    symmetry.add_site_family(lat.sublattices[1], other_vectors=[(-1,2)])

    lead_0 = kwant.Builder(symmetry)
    lead_0[lat.shape(lead_shape, (0,0))] = V
    lead_0[lat.neighbors()] = -t

    lead_0[A(-N, 2*N), B(0, 0)] = -t * np.exp(-1j*phi)
    lead_1 = lead_0.reversed()
    return [lead_0, lead_1]

def W_to_close(W):
    N = max(W // np.sqrt(3), 2)
    return N * np.sqrt(3), int(N)

def assemble_syst(syst, leads):
    for lead in leads:
        syst.attach_lead(lead)
    return syst.finalized()

def calculate_bands(lead, Npts):
    bands = kwant.physics.Bands(lead.finalized())
    momenta = np.linspace(-np.pi, np.pi, Npts)
    energies = [bands(k) for k in momenta]
    return momenta, energies

def plot_bands(lead, Npts=201):
    momenta, bands = calculate_bands(lead, Npts)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(momenta, bands, color='k')
    plt.tight_layout()
    plt.show()

def calculate_DOS(fsystem):
    spectrum = kwant.kpm.SpectralDensity(fsystem)
    energies, densities = spectrum()
    return energies, densities

def plot_DOS(fsystem):
    energies, densities = calculate_DOS(fsystem)
    figure, ax = plt.subplots(figsize=(7,5))
    # ax.plot(energies, densities.real)
    ax.plot(energies, densities.real, lw=2)
    ax.set_xlim(-8,8)
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('DOS [a.u.]')
    plt.tight_layout()
    plt.show()

def calculate_LDOS(fsyst, Energy):
    kwant_op = kwant.operator.Density(fsyst, sum=False)
    local_dos = kwant.kpm.SpectralDensity(fsyst, operator=kwant_op)
    local_dos.add_vectors(100)
    zero_energy_ldos = local_dos(energy=0)
    finite_energy_ldos = local_dos(energy=Energy)
    return zero_energy_ldos, finite_energy_ldos

def plot_LDOS(fsystem, Energy=1.0):
    zero_energy_ldos, finite_energy_ldos = calculate_LDOS(fsystem, Energy)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    kwant.plotter.density(fsystem, zero_energy_ldos.real, ax=axes[0], cmap='inferno')
    kwant.plotter.density(fsystem, finite_energy_ldos.real, ax=axes[1], cmap='inferno')
    axes[1].get_xaxis().set_ticklabels([])
    axes[1].get_yaxis().set_ticklabels([])
    # axes[0].text(0.05, 0.6,'(a)', color='w',
    #            fontsize=30, fontname='Arial',
    #            horizontalalignment='center', verticalalignment='top',
    #            transform = axes[0].transAxes)
    # axes[1].text(0.05, 0.6,'(b)',
    #            color='k', fontsize=30,
    #            fontname='Arial', horizontalalignment='center',
    #            verticalalignment='top', transform = axes[1].transAxes)
    axes[0].tick_params(labelsize=20)
    axes[0].set_xlabel('x [a]', fontsize=20)
    axes[0].set_ylabel('y [a]', fontsize=20)
    #
    # ldos_map1, = axes[0].images
    # ldos_map2, = axes[1].images

    # divider1 = make_axes_locatable(axes[0])
    # cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    #
    # divider2 = make_axes_locatable(axes[1])
    # cax2 = divider2.append_axes("right", size="2%", pad=0.05)

    # plt.colorbar(ldos_map1, cax=cax1, ticks=[.0, .6, 1.1])
    # plt.colorbar(ldos_map2, cax=cax2,)
    plt.tight_layout()
    # plt.savefig('ldos_E_0_and_E_1_without_adatom.png', dpi=200)
    plt.show()

def wave_function(fsyst):
    # Plot wave function squared for the first mode for a specified energy and ky
    wf = kwant.solvers.default.wave_function(fsyst, energy=0.15)
    # wf = kwant.solvers.default.wave_function(fsyst, energy=2.0, params=dict(k_x=None, k_y=ky))
    psi = wf(0)[0]
    max_val_dens = np.max(np.abs(psi)**2)
    min_val_dens = np.min(np.abs(psi)**2)


    min_real = np.min(psi.real)
    max_real = np.max(psi.real)

    # print('\t MINIMUM VALUE OF PSI.REAL = ', min_real)
    # print('\t MAXIMUM VALUE OF PSI.REAL = ', max_real)

    fig, ax = plt.subplots(ncols=3, figsize=(12,6))
    kwant.plotter.map(fsyst, np.real(psi), ax=ax[0], show=False)
    kwant.plotter.map(fsyst, np.imag(psi), ax=ax[1], show=False)
    kwant.plotter.map(fsyst, np.abs(psi)**2, ax=ax[2], show=False, vmax=max_val_dens, vmin=0)
    plt.show()

def main():
    system, leads = make_syst_and_leads(W=20, L=20, phi=0.5)
    fsyst = assemble_syst(system, leads)
    # kwant.plot(fsyst)
    # plot_bands(leads[0])
    # plot_DOS(fsyst)
    # wave_function(fsyst)
    # plot_LDOS(fsyst, 0.5)

if __name__ == '__main__':
    main()
