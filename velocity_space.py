'''Utility functions relating to computing subgroups in velocity space

This module contains functions for computing subgroups in velocity space,
as well as analysing their dispersion. Plotting utilities for the subgroups
can be found in plotting_utils.

Sofie L
11/05/2022
'''

import hdbscan
import vaex, vaex.ml
import numpy as np

########################################################################################################        
        
def apply_HDBSCAN_vspace(df):
    ''''
    Applies HBSCAN to the the vR, vphi and vz-components of each cluster in df.
    Inserts an addition column describing subgroup membership.
    The parameters are set according to
    min_cluster_size = max(3, int(0.05*stars.count())
                      (a subgroup needs to contain at least 5% of the stars of a cluster, 
                       but at the minimum three stars)
    min_samples = 1 (resulting in being lenient on possible noise)
    allow_single_cluster = True if and only if the standard deviation in at least two polar velocity
                           velocity components is small (<25km/s). This is a heuristic check to see if
                           a cluster seems to be a single blob or string.
    
    Parameters:
    df(vaex.DataFrame): Input dataframe containing cluster labels and velocity components
                      
    Returns:
    df(vaex.DataFrame): Input dataframe with an additional column indicating velocity space
                        subgroup membership as a positive integer. Zero indicates 
                        that the star is not in a subgroup.
    
    '''
    v_space_labels = np.zeros(df.count()) * np.nan
    
    #Loop over clusters and apply HDBSCAN separately to each one of them.
    for i in range(1, int(max(np.unique(df['labels'].values)+1))): 

        stars = df[df[labelcol]==i]
        N_members = stars.count()

        X = stars['vR', 'vphi', 'vz'].values
        
        single_cluster_bool = False
        stds_1D = np.array([np.std(stars.vR.values), np.std(stars.vphi.values), np.std(stars.vz.values)])
        if (sum(stds_1D < 30) >= 2):
            single_cluster_bool = True
        
        #proposed approach
        clusterer = hdbscan.HDBSCAN(allow_single_cluster = single_cluster_bool,
                                    min_cluster_size = max(3, int(0.05*stars.count())),
                                    min_samples = 1)
        
        clusterer.fit(X)    

        v_space_labels[np.where(df['labels'].values==i)] = clusterer.labels_
    
    #HDBSCAN gives -1 for no cluster membership, but we put 0 for conformity with IOM labels
    v_space_labels = v_space_labels + 1
    v_space_labels = np.nan_to_num(v_space_labels)
    
    if('substructure' in labelcol):
        df['v_space_labels_substructure'] = v_space_labels.astype(int)
    else:
        df['v_space_labels'] = v_space_labels.astype(int)
        
    return df

########################################################################################################   
