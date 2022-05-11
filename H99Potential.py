import numpy as np

################################################################################
### ORIGINAL FILE BY JOVAN VELJANOSKI
################################################################################
def potential_halo(x, y, z,):
    '''
    Calculates the potential contribution coming fromt the halo.
    '''
    G = 4.301*10**-6   # fixed GMsun [kpc km^2/s^2]

    # parameters
    M200 = 1e12
    rs = 21.5
    Rvir = 258
    c = Rvir/rs


    #halculate the halo potential
    phi_0 = G*M200/Rvir /(np.log(1+c)-c/(1+c))*c
    r = np.sqrt(x*x + y*y + z*z)
    
    return - phi_0 * rs/r * np.log(1.0 + r/rs) 

################################################################################
def potential_disk(x,y,z):
    '''
    Calculates the potential due to the disk
    '''
    # parameters
    a = 6.5
    b = 0.26
    G = 4.301*10**-6   # fixed GMsun [kpc km^2/s^2]
    Mdisk = 9.3*10**+10
    GMd = G * Mdisk

    sqd = np.sqrt(z**2.0 +b**2.0)

    # square root of the density, probably
    sqden1 = np.sqrt(x**2. + y**2.0 + (a+sqd)**2.0)

    # the potential of the disk
    phi_d = -GMd/sqden1

    return phi_d

################################################################################
def potential_bulge(x, y, z):
    '''
    Calculates the potential contribution due to the bulge
    '''
    # parameters
    c = 0.7
    Mbulge = 3.0*10**+10
    G = 4.301*10**-6   # fixed GMsun [kpc km^2/s^2]
    GMb = G * Mbulge

    # radial distance
    r = np.sqrt(x**2. + y**2. + z**2.)

    # the potential due to the bulge
    phi_b = -GMb/(r+c)

    return phi_b

################################################################################
def potential_full(x, y, z, verbose=False):
    '''
    Calculates the potential energy of a star given (x,y,z) coordinates,
    centred on the Sun, via those Aminas FORTRAN functions that
    I've translated here.

    (x,y,z) are assumed to be in kpc
    '''

    ### Compute the potentials due to the different components

    # The disk contribution
    phi_disk = potential_disk(x=x, y=y, z=z)
    # The bulge contribution
    phi_bulge = potential_bulge(x=x, y=y, z=z)
    # The halo contribution
    phi_halo = potential_halo(x=x, y=y, z=z)


    # the sum of all these
    phi_total = phi_disk + phi_bulge + phi_halo

    if verbose:
        print(phi_total,phi_disk, phi_bulge, phi_halo)

    # Finally, return the total potential
    return phi_total

################################################################################
def En_L_calc(x,y,z,vx,vy,vz,):
    '''
    Assumes the coordinates are centred on the Sun.
    Units are in kpc, and km/s.
    '''

    # Shift the to Galactic centre, and correct for the LSR
    x = x -8.2
    vx = vx + 11.1
    vy = vy + 232.8 + 12.24
    vz = vz + 7.25

    # The angular momentum components
    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx

    # The total Angular momentum
    Ltotal = np.sqrt(Lx**2. + Ly**2. + Lz**2.)

    # The perpendicular component
    Lperp = np.sqrt(Ltotal**2. - Lz**2.0)

    # The energy
    En = (vx**2. + vy**2. + vz**2.)/2. + potential_full(x,y,z)

    return En, Ltotal, Lz, Lperp

################################################################################

from numpy import sqrt,log
def vc_halo(x, y, z,):
    '''
    Calculates the potential contribution coming fromt the halo.
    '''
    G = 4.301*10**-6   # fixed GMsun [kpc km^2/s^2]

    # parameters
    M200 = 1e12
    rs = 21.5
    Rvir = 258
    c = Rvir/rs


    #halculate the halo potential
    phi_0 = G*M200/Rvir /(np.log(1+c)-c/(1+c))*c
    r = np.sqrt(x*x + y*y + z*z)
    
    return np.sqrt(r*( phi_0 * rs/r * (log(1 + r/rs)/r - 1/(rs+r))  ))


def vc_disk(x,y,z):
    '''
    Calculates the potential due to the disk
    '''
    # parameters
    a = 6.5
    b = 0.26
    G = 4.301*10**-6   # fixed GMsun [kpc km^2/s^2]
    Mdisk = 9.3*10**+10
    GMd = G * Mdisk

    sqd = np.sqrt(z**2.0 +b**2.0)

    # square root of the density, probably
    sqden1 = np.sqrt(x**2. + y**2.0 + (a+sqd)**2.0)

    # the potential of the disk
    vc_d = (x*x+y*y)*(GMd/sqden1**3)

    return sqrt(vc_d)

def vc_bulge(x, y, z):
    '''
    Calculates the potential contribution due to the bulge
    '''
    # parameters
    c = 0.7
    Mbulge = 3.0*10**+10
    G = 4.301*10**-6   # fixed GMsun [kpc km^2/s^2]
    GMb = G * Mbulge

    # radial distance
    r = np.sqrt(x**2. + y**2. + z**2.)

    # the potential due to the bulge
    vc_b = GMb*r/(r+c)**2

    return sqrt(vc_b)

def vc_full(x, y, z, verbose=False):
    '''
    Calculates the potential energy of a star given (x,y,z) coordinates,
    centred on the Sun, via those Aminas FORTRAN functions that
    I've translated here.

    (x,y,z) are assumed to be in kpc
    '''

    ### Compute the potentials due to the different components

    # The disk contribution
    vcf_disk = vc_disk(x=x, y=y, z=z)
    # The bulge contribution
    vcf_bulge = vc_bulge(x=x, y=y, z=z)
    # The halo contribution
    vcf_halo = vc_halo(x=x, y=y, z=z)


    # the sum of all these
    vc_total = np.sqrt(vcf_disk**2 + vcf_bulge**2 + vcf_halo**2)

    if verbose:
        print(vc_total,vcf_disk, vcf_bulge, vcf_halo)

    # Finally, return the total potential
    return vc_total