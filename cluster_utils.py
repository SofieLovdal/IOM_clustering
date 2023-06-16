'''Clustering Utility Functions

Contains helper functions for the IOM-clustering algorithm.

Sofie L
21/10/2020 v1
26/02/2021 v2
28/5/2021 v3
'''

import vaex, vaex.ml
import numpy as np
import H99Potential as Potential
import matplotlib
import matplotlib.pyplot as plt
from vaex.ml import MinMaxScaler
from scipy.cluster.hierarchy import linkage
from tqdm.notebook import tqdm
from queue import Queue
np.random.seed(0)

####################################################################################################

def get_artificial_dataset(ds):
    '''
    Returns an artificial dataset by shuffling the vy and vz-components of the original dataset.
    The artificial dataset is cropped to have the exact same number of halo-members as the original.
    
    Parameters:
    ds(vaex.DataFrame): The original dataset (GAIA RSV-halo-sample).
    
    Returns:
    df_art: A dataframe containing the artificial dataset.
    '''
    vy_scrambled = ds.vy.values
    np.random.shuffle(vy_scrambled)

    vz_scrambled = ds.vz.values
    np.random.shuffle(vz_scrambled)

    En_art, Ltotal_art, Lz_art, Lperp_art = Potential.En_L_calc(*ds.evaluate(['x','y','z','vx']),vy_scrambled, vz_scrambled)
    
    _vy_scrambled = (vy_scrambled+12.24)-ds.variables['vlsr']*np.cos(ds.evaluate('rPhi'))
    _vz_scrambled = vz_scrambled+7.25
    vtoomre = np.sqrt(ds._vx**2 + (_vy_scrambled-232)**2 + _vz_scrambled**2)
    
    vtoomre= np.sqrt(ds._vx.values**2 + (_vy_scrambled-232)**2 + _vz_scrambled**2)
    df_art = vaex.from_arrays(vx=ds.vx.values, vy=vy_scrambled, vz=vz_scrambled, 
                              En=En_art, Lz=-Lz_art, Ltotal=Ltotal_art, Lperp=Lperp_art,
                              vtoomre=vtoomre)
    
    df_art=df_art[(df_art.vtoomre>210) & (df_art.En<0)]
    
    num_stars = ds[(ds.vtoomre>210) & (ds.En<0)].count()
    
    if(df_art.count()>num_stars):
        df_art = df_art.sample(num_stars)
    
    return df_art

####################################################################################################

def scaleData(ds, features, minmax_values):
    '''
    Scales data to the range [-1, 1].
    
    Parameters:
    ds(vaex.DataFrame): Dataframe containing the data to be scaled
    features([str]): List of features to be scaled
    minmax_values(vaex.DataFrame): Dataframe containing reference minimum and maximum 
                                     values for the scaling of the data (necessary in order to have 
                                     a fixed reference origin)
    
    Returns:
    ds(vaex.DataFrame): The original dataframe with appended columns containing the scaled features
    '''
    scaler = vaex.ml.MinMaxScaler(feature_range=[-1, 1], features=features, prefix='scaled_')
    scaler.fit(minmax_values)
    
    return scaler.transform(ds)

####################################################################################################

def get_artificial_set(N, ds):
    '''
    Creates a data set of N artificial halos
    
    Parameters:
    N(int): The number of artificial halos to be generated
    ds(vaex.DataFrame): A dataframe containing the halo stars and a little bit more, recommended: V-V_{lsr}>180.
    
    Returns:
    df_art_all(vaex.DataFrame): A dataframe containing each artificial halo, their integrals of motion, and data set index.
    '''
    temp=0
    print(f'Creating {N} artificial datasets...')
    for n in tqdm(range(N)):
        
        df_art = get_artificial_dataset(ds)
        df_art['index'] = np.array(int(df_art.count())*[n])
        
        if(temp==0):
            df_art_all = df_art
            temp=1
        else:    
            df_art_all=df_art_all.concat(df_art)
        
    return df_art_all

####################################################################################################

def clusterData(df, features, linkage_method):
    '''
    Scale necessary features and apply hierarchical clustering.
    Generate linkage matrix based on data in df. Returns the linkage matrix Z.
    
    Parameters:
    df(vaex.DataFrame): Dataframe with columns to be used in clustering
    features([str]): Names of the clustering features
    linkage_method(str): linkage method to be used in clustering
    
    Returns:
    Z(np.ndarray): The linkage matrix resulting from hierarchical clustering
    '''

    X = df[features].values
    Z = linkage(X, linkage_method)
    
    return Z

####################################################################################################

def get_members(i, Z):
    '''
    Returns the list of members for a specific cluster included in the encoded linkage matrix
    
    Parameters:
    i(int): candidate cluster number
    Z(np.ndarray): the linkage matrix
    
    Returns:
    members(np.array): array of members corresponding to candidate cluster i
    '''
    members = []
    N = len(Z) + 1 #the number of original data points
    
    # Initializing a queue
    q = Queue(maxsize = N)
    q.put(i)
    
    while(not q.empty()):
        
        index = q.get()
        
        if(Z[index, 0]<N): 
            members.append(int(Z[index, 0])) #append original member
        else:
            q.put(int(Z[index, 0]-N)) #backtract one step, enqueue this branch
            
        if(Z[index,1]<N):
            members.append(int(Z[index,1])) #append original member
        else:
            q.put(int(Z[index, 1]-N)) #enqueue this branch
    
    return np.array(members)

######################################################################################################

def get_cluster_labels_with_significance(significant, Z):
    '''
    Extracts flat cluster labels given a list of statistically significant clusters 
    in the linkage matrix Z, and also returns a list of their correspondning statistical significance.
    
    Parameters:
    significant(vaex.DataFrame): Dataframe containing statistics of the significant clusters.
    Z(np.ndarray): The linkage matrix outputted from single linkage
    
    Returns:
    labels(np.array): Array matching the indices of stars in the data set, where label 0 means noise 
                      and other natural numbers encode significant structures.
    significance_list(np.array): The statistical significance corresponding to each entry in labels
    '''
    
    print("Extracting labels...")
    labels = np.zeros(len(Z)+1)
    significance_list = np.zeros(len(Z)+1)
    
    label_count = 1
    
    i_list = np.sort(significant.i.values)

    for cluster in tqdm(i_list):
        
        members = get_members(cluster, Z)

        current_significance = significant[significant.i == cluster].significance.values[0]

        labels[members.astype(int)]=label_count
        significance_list[members.astype(int)] = current_significance
        label_count = label_count+1

    unique_labels = np.unique(labels)
    
    print(f'Number of clusters: {len(unique_labels)-1}')
    
    return labels, significance_list

######################################################################################################

def select_maxsig_clusters_from_tree(stats, minimum_significance=3):
    
    '''
    Takes all (possibly hierarchically overlapping) significant clusters from the
    single linkage merging process and determines where each nested structure reaches its
    maximum statistical significance in the dendrogram (or merging tree). Returns the indices
    of these maximum significance clusters such that we can extract flat cluster labels 
    for the star catalogue.
    
    Parameters:
    stats(vaex.DataFrame): Dataframe containing information from the single linkage merging process.
    minimum_significance: The smallest significance (sigma) that we accept for a cluster
    
    Returns: 
    selected(np.array): The indices (i in stats) of the clusters that have been selected from the tree
    max_sign(np.array): The statistical significance of each selected cluster
    
    '''
    print('Picking out the clusters with maximum significance from the tree...')
    
    significant = stats[stats.significance > minimum_significance]
    max_i = np.max(significant.i.values)
    N = stats.count()+1

    selected_i = 0

    traversed = [] #list of indices i that we have already traversed
    selected = [] #list of merges in the tree where the significance is maximized
    max_sign = [] #the significance at the selected step

    array = significant['significance', 'i'].values
    array = np.flip(array[array[:, 0].argsort()]) #Sort according to significance

    for merge_idx in tqdm(array[:, 0]): #Traverse over the significant clusters, starting with the most significant ones.

            traversed_currentpath = []
            maxval = 0
            row = significant[significant.i == merge_idx]
            i = row.i.values[0]
            significance = row.significance.values[0]
            visited = 0

            #traverse up the whole tree
            while(i <= max_i):

                #only go up the tree, in case we have not already examined this path and found that the
                #max significance was already found at a lower level.
                if(i not in traversed):

                    traversed_currentpath.append(i) #Track what we have traversed since the max-significance

                    if(significance > maxval):
                        maxval = significance
                        selected_i = i
                        traversed_currentpath = []

                    next_cluster = stats[(stats.index1 == i+N) | (stats.index2 == i+N)]
                    i = next_cluster.i.values[0]
                    significance = next_cluster.significance.values[0]
                    visited = 1

                else: #we already traversed this path, so exit
                    break

            if((maxval > 0) and (visited == 1)): #It means that there was some part of this path which was not yet traversed.
                traversed.extend(traversed_currentpath)
                
                if(selected_i not in selected): #Same cluster might have been marked the most significant one for multiple paths.
                    selected.append(selected_i)
                    max_sign.append(maxval)
                    
    return np.array(selected), np.array(max_sign)               


###########################################################################################################

def count_stars(df, df_artificial, stats, Z, min_members, max_members, features, n_std, N_datasets):
    '''
    The function applies PCA to each candidate cluster C included in the linkage matrix Z.
    We then define an ellipsoidal boundary around C, where the axes lengths of the ellipsoid 
    are chosen to be n standard deviations of spread along each axis (n=N_sigma_ellipse_axis).
    We then map stars from the artificial halo to the PCA-space defined by C and use the standard
    equation of an ellipse to check how many artificial halo stars fall within the ellipsoid defined by C.
    This is also done for the stars of cluster C itself.
    
    Parameters:
    df(vaex.DataFrame): Dataframe containing clustering features
    df(vaex.DataFrame): Dataframe representing smooth background
    stats(vaex.DataFame): Dataframe containing some information
    Z(np.ndarray): Linkage matrix outputted from the single linkage function
    
    Returns:
    art_region_count(np.ndarray): Normalized number of stars within the boundary of 
                                  each candidate cluster for the artificial data sets
    art_region_count_std(np.ndarray): Standard deviation over the counts for each
                                      artificial data set realization
    real_region_count(np.ndarray): Number of stars in the real data set falling within
                                   the cluster boundary, for each candidate cluster
    '''
    
    art_region_count = np.zeros(stats.count())
    art_region_count_std = np.zeros(stats.count())
    real_region_count = np.zeros(stats.count())
    
    for i in tqdm(stats.i.values):
        
        #get a list of members (indices in our halo set) of the cluster C we are currently investigating
        members = get_members(i, Z)

        #ignore these clusters, return placeholder
        if((len(members)>max_members) or (len(members)<min_members)):
            art_region_count[i] = len(members)
            art_region_count_std[i] = 1
            real_region_count[i] = 1
        
        else:
            #Get a dataframe containing the members
            df_members = df.take(np.array(members))

            #Fit a PCA-object according to the members of cluster C
            pca = vaex.ml.PCA(features=features, n_components=len(features))
            pca.fit(df_members)

            [[En_lower, En_upper], [Lperp_lower, Lperp_upper], [Lz_lower, Lz_upper]] = \
            df_members.minmax(features)

            eps = 0.05 #large enough such that ellipse fits within

            #Extract neighborhood of C, so that we do not have to PCA-map the full artificial dataset  
            region = df_artificial[(df_artificial.scaled_En>En_lower-eps) & (df_artificial.scaled_En<En_upper+eps) & 
                          (df_artificial.scaled_Lperp>Lperp_lower-eps) & (df_artificial.scaled_Lperp<Lperp_upper+eps) &
                          (df_artificial.scaled_Lz>Lz_lower-eps) & (df_artificial.scaled_Lz<Lz_upper+eps)]

            #Map the stars in the artificial data set to the PCA-space defined by C
            region = pca.transform(region)

            r0 = n_std*np.sqrt(pca.eigen_values_[0])
            r1 = n_std*np.sqrt(pca.eigen_values_[1])
            r2 = n_std*np.sqrt(pca.eigen_values_[2])

            #Extract the artificial halo stars that reside within the ellipsoid
            within_region = region[((np.power(region.PCA_0, 2))/(np.power(r0, 2)) + 
                                   (np.power(region.PCA_1, 2))/(np.power(r1, 2)) +
                                   (np.power(region.PCA_2, 2))/(np.power(r2, 2)))<=1]

            #Average count and standard deviation in this region.
            #Vaex takes limits as non-inclusive in count(), so we add a small value to the default minmax limits
            region_count = within_region.count()/N_datasets
            counts_per_halo = within_region.count(binby="index", limits=[-0.01, N_datasets-0.99], shape=N_datasets)
            region_count_std = np.std(counts_per_halo)

            art_region_count[i] = region_count
            art_region_count_std[i] = region_count_std

            '''Extract the same value for the real data set'''
            #Extract neighborhood of C, so that we do not have to PCA-map the full artificial dataset  
            region = df[(df.scaled_En>En_lower-eps) & (df.scaled_En<En_upper+eps) & 
                        (df.scaled_Lperp>Lperp_lower-eps) & (df.scaled_Lperp<Lperp_upper+eps) &
                        (df.scaled_Lz>Lz_lower-eps) & (df.scaled_Lz<Lz_upper+eps)]

            #Map the stars in the artificial data set to the PCA-space defined by C
            region = pca.transform(region)

            #Extract the artificial halo stars that reside within the ellipsoid
            within_region = region[((np.power(region.PCA_0, 2))/(np.power(r0, 2)) + 
                                   (np.power(region.PCA_1, 2))/(np.power(r1, 2)) +
                                   (np.power(region.PCA_2, 2))/(np.power(r2, 2)))<=1]

            region_count = within_region.count()
            real_region_count[i] = region_count

    return art_region_count, art_region_count_std, real_region_count

###########################################################################################################
