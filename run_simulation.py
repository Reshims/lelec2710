import kwant
import scipy
import numpy as np
from multiprocessing import Pool
from time import time

from noise import pnoise2
from tqdm import tqdm
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt

# CONSTANTS
h = 6.626069e-34    # Planck constant (Js)
hbar = h / (2*np.pi)
e = 1.602e-19      # electron charge (C)
c = 299792458       # Light celerity (m/s)
kB = 1.3806503e-23  # Boltzmann constant (kg m2/Ks2)
m_e = 9.1e-31       # electron mass (kg)

# PARAMS
W = .72e-6
r1 = .95e-6
r2 = 1.54e-6
asquare = 1e-9
tsquare = 1

xmax = 2.62e-6

"""scaling"""
scaling = 12 # scaling factor
a = asquare*scaling
t = tsquare/(scaling**2)

l_pot = 150e-9
w_pot = 150e-9
def sys_builder(a=1,t=1,W=10,r1=10,r2=20, max_x=25, Vg=0, xpos=0, ypos=0, Rp=w_pot):
    lat = kwant.lattice.square(a, norbs = 1) # norbs gives the number of orbitals per atom. It is needed to compute J
    sys = kwant.Builder() # construction of the system

    #helper functions
    def geom(pos):
        x, y = pos
        rsq = x**2+y**2
        minr_cond = r1**2<rsq
        maxr_cond = rsq<r2**2
        exrect_cond = (-W/2 < y < W/2) and (-max_x < x)

        #we're drawing two shapes: 1) a circle with a bar in its center
        #                          2) an horizontal bar
        #
        #condition to be in shape: 1) radius < max_radius and (radius > min_radius or |x| < bar_width/2)
        #                          2) radius > min_radius and |y| < bar_height/2 and |x| < max_x
        return ((minr_cond and maxr_cond) or (exrect_cond and minr_cond)) and (x <= 0)

    def onsite(site, xpos, ypos, Rp):
        x, y = site.pos

        return 4*t + Vg*elec_pot(site, xpos, ypos, Rp) + disorder(site)

    def elec_pot(site, xpos, ypos, Rp):
        x, y = site.pos
        xp, yp = (x-xpos)/Rp, (y-ypos)/Rp

        return 1 / ((1+xp*xp)*(1+yp*yp)*(1+(xp-yp)**2)*(1+(xp+yp)**2))

    def disorder(site):
        x, y = site.pos

        pot = pnoise2(20*x/xmax+30.099439158876095, 20*y/xmax+19.71041433162528, octaves=10, persistence=.4, lacunarity=5)
        return pot*t/10

    #specify onsite energy of the system (hopping will be done later)
    sys[lat.shape(geom, (0,r1+a))] = lambda site: onsite(site, xpos, ypos, Rp)

    #leads definiton
    sym_lead_left = kwant.TranslationalSymmetry((-a, 0))
    lead_left = kwant.Builder(sym_lead_left)

    sym_lead_right = kwant.TranslationalSymmetry((a, 0))
    lead_right = kwant.Builder(sym_lead_right)

    def lead_shape_left(pos):
        (x, y) = pos
        return (-W / 2 < y < W / 2)

    lead_left[lat.shape(lead_shape_left, (0, 0))] = 4 * t
    lead_left[lat.neighbors()] = -t

    def lead_shape_right(pos):
        (x, y) = pos
        return (r1 < abs(y) < r2)

    #specify onsite energy of the leads
    lead_right[lat.shape(lead_shape_right, (0, r1*1.2))] = 4 * t
    lead_right[lat.shape(lead_shape_right, (0, -r1*1.2))] = 4 * t

    #specify onsite energy
    lead_right[lat.neighbors()] = -t


    sys.attach_lead(lead_left)
    sys.attach_lead(lead_right)

    return sys.finalized()

#do the heavy lifting, ie. the simulation for one set of parameters
def compute_transmission(Ef, Vg, xpos, ypos, Rp):
    sys = sys_builder(a,t,W,r1,r2,xmax, Vg, xpos, ypos, Rp)
    smatrix = kwant.smatrix(sys, energy=Ef)

    return smatrix.transmission(1, 0)

#imap_unordered helper function
def map_compute(args):
    return compute_transmission(*args)

#interface for main
#xpos has the dimension of the changing parameters
#
#eg: simulation of N different Rp and M different Ef
#    => shape(xpos) = shape(Rp) = shape(Ef) = NxM (or MxN)
#    => all other parameters have this shape or are scalars
def get_transmission(xpos, ypos, Ef=t/2, Vg=.9*t/2, Rp=w_pot, filename=''):
    N = xpos.size

    args = np.zeros((N, 5))
    args[:, 0] = Ef   if len(Ef) > 1 else Ef[0]
    args[:, 1] = Vg   if len(Vg) > 1 else Vg[0]
    args[:, 2] = xpos if len(xpos) > 1 else xpos[0]
    args[:, 3] = ypos if len(ypos) > 1 else ypos[0]
    args[:, 4] = Rp   if len(Rp) > 1 else Rp[0]

    outputs = []
    pbar = tqdm(total=N, desc=filename) #loadbar

    with Pool(processes=4) as pool:
        for result in pool.imap_unordered(map_compute, args):
            outputs.append(result)
            pbar.update(1)

    return np.array(outputs)

#adaptative mesh for 2D conductance
#this will generate a full grid if w_over=inf, r1_under=0 and r2_over=inf
def get_positions(N=10, M=10, w_over=(1 + 2*w_pot/W), r1_under=(1 - w_pot/r1), r2_over=(1 + w_pot/r2), W=W, r1=r1, r2=r2, max_x=xmax):
    def geom(x, y):
        rsq = x**2+y**2
        minr_cond = (r1*r1_under)**2 <= rsq
        maxr_cond = rsq < (r2*r2_over)**2
        exrect_cond = (abs(y) < W*w_over/2) & (-max_x < x)

        return ((minr_cond & maxr_cond) | (exrect_cond & minr_cond)) & (x <= 0)

    x = np.linspace(-(2*r2 - r1), 0, N)
    y = np.linspace(0, r2, M)
    xx, yy = np.meshgrid(x, y)
    truths = geom(xx, yy)

    return xx, yy, truths

#interface for batch
#truths will select at which xx & yy to simulate conductance
#all the other positions will have G0 (conductance with Vg=0)
def main(xx, yy, truths, Vg=.9*t/2, Ef=t/2, Rp=w_pot, filename='data.mat'):
    T_pos = get_transmission(xx[truths].flatten(), yy[truths].flatten(), \
                             Ef=np.array(Ef).flatten(), Vg=np.array(Vg).flatten(), \
                             Rp=np.array(Rp).flatten(), filename=filename.split('/')[-1][:-4])

    data = np.zeros_like(xx)

    data[truths] = T_pos
    data[~truths] = compute_transmission(t/2, 0, r2, r2, w_pot)

    savemat(filename, dict(xx=xx, yy=yy, truths=truths, Ef=Ef, Vg=Vg, Rp=Rp, data=data))


if __name__ == '__main__':
    if True: #plot domain
        wfac = (1 + 2*w_pot/W) #* 100
        r1fac = (1 - w_pot/r1) #* 0
        r2fac = (1 + w_pot/r2) #* 100
        N = 400
        M = 200

        xx, yy, dtrh = get_positions(N, M, r2_over=r2fac, w_over=wfac, r1_under=r1fac)
        _, _, reals = get_positions(N, M)

        plt.scatter(xx[reals]/r2, yy[reals]/r2, marker='.', color='tab:green')
        plt.scatter(xx[dtrh & (~reals)]/r2, yy[dtrh & (~reals)]/r2, marker='.', color='tab:blue')
        plt.scatter(xx[~dtrh]/r2, yy[~dtrh]/r2, marker='.', color='tab:red')

        print(100*sum(dtrh.flatten())/(N*M))
        plt.show()

    if False: #plot current
        sys = sys_builder(a,t,W,r1,r2,xmax, -.9*t/2, -r1, 0, w_pot)
        wfs = kwant.wave_function(sys, energy=Ef)
        scattering_wf = wfs(0)  # all scattering wave functions from lead 0
        J0 = kwant.operator.Current(sys)
        wf_left = wfs(0)
        current = J0(wf_left[0]) # to sum over all the lead's mode

        kwant.plotter.current(sys, current, cmap='viridis')
