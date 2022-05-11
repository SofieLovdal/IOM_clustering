'''
This module takes a Gaia-style catalogue and computes integrals of motion and 
a few other derived quantities based on the raw data.
'''

import vaex
import numpy as np
import H99Potential as Potential

#############################################################################################################################

def get_GAIA(ds):
    '''
    Computes integrals of motion for stars in the input catalogue and 
    adds these as columns to the dataframe.
    
    Parameters:
    ds(vaex.DataFrame): A dataframe containing heliocentric xyz-coordinates (kpc)
                        and xyz-velocity components (km/s).
                        
    Returns:
    ds(vaex.DataFrame): The input dataframe with added columns of total energy, 
                        angular momentum, circularity, together with positions and 
                        velocity components in polar coordinates.
    '''

    # Motion of the LSR, and solar position w.r.t. Galactic centre (McMillan (2017))
    (vlsr, R0) = (232.8, 8.2)
    # Solar motion w.r.t. LSR (Schoenrich et al. (2010))
    (_U,_V,_W) = (11.1, 12.24, 7.25)

    # Adding these as variables to the dataframe
    ds.add_variable('vlsr',232.8)
    ds.add_variable('R0',8.20)
    ds.add_variable('_U',11.1)
    ds.add_variable('_V',12.24)
    ds.add_variable('_W',7.25)

    # Compute energy and angular momentum components using the potential used in 
    # Koppelman et al. (2019): Characterization and history of the Helmi streams with Gaia DR2
    En, Ltotal, Lz, Lperp = Potential.En_L_calc(*ds.evaluate(['x','y','z','vx','vy','vz']))

    # Adding the columns to the data frame
    ds.add_column('En',En)
    ds.add_column('Lz',-Lz) ## Note: sign is flipped (just a convention)
    ds.add_column('Lperp',Lperp)
    ds.add_column('Ltotal', Ltotal)

    # Adding some more columns (Galactocentric Cartesian velocities)
    ds.add_virtual_column('vx_apex',f'vx+{_U}')
    ds.add_virtual_column('vy_apex',f'vy+{_V}+{vlsr}')
    ds.add_virtual_column('vz_apex',f'vz+{_W}')
    
    # Polar coordinates
    ds.add_virtual_columns_cartesian_to_polar(x='(x-8.2)', radius_out='R', azimuth_out='rPhi',radians=True)

    # Velocities in polar coordinates
    ds.add_virtual_columns_cartesian_velocities_to_polar(x='(x-8.2)', vx='vx_apex', vy='vy_apex',vr_out='vR', vazimuth_out='_vphi',propagate_uncertainties=False)  
    ds.add_virtual_column('vphi','-_vphi')  # flip sign for convention
    
    # position depending correction for the LSR
    ds.add_column('_vx', ds.vx.values+11.1-ds.variables['vlsr']*np.sin(ds.evaluate('rPhi')))
    ds.add_column('_vy', ds.vy.values+12.24-ds.variables['vlsr']*np.cos(ds.evaluate('rPhi')))
    ds.add_column('_vz', ds.vz.values+7.25)
        
    # The selection
    ds.select('(_vx**2 + (_vy-232)**2 + _vz**2) > 210**2', name='halo')

    # Toomre velocity: velocity offset from being a disk orbit
    ds.add_virtual_column('vtoomre','sqrt(_vx**2 + (_vy-232)**2 + _vz**2)')
    
    # The potential may compute energies larger than 0 in some edge cases, these are filtered out.
    return ds[(ds.En<0)]
