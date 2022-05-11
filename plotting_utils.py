'''Utility functions for plotting

Contains helper functions visualization of the clustering results, both in integrals-of-motion 
space and in velocity space.

Sofie L
09/05/2022
'''

import vaex
import numpy as np

import matplotlib
from matplotlib import colors
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable as sm

mpl.rcParams['figure.dpi'] = 150
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

####################################################################################################

def get_cmap(df_minsig):
    '''
    Returns a cmap and a norm which maps the three sigma labels to proper color bins.
    The last few colors of the colormap are very light, so we replace them for better 
    visible colors.
    
    
    Parameters:
    df_minsig(vaex.DataFrame): The slice of the dataframe containing only the relevant clusters.
    
    Returns:
    cmap(matplotlib.colors.Colormap): The colormap we use for plotting clusters
    norm(matplotlib.colors.Normalize): Normalization bins such that unevenly spaced
                                       label values will get mapped
                                       to linearly spaced color bins
    '''
    unique_labels = np.unique(df_minsig.labels.values)
    cmap1 = plt.get_cmap('gist_ncar', len(unique_labels))
    cmaplist = [cmap1(i) for i in range(cmap1.N)]
    
    if(len(unique_labels)>7):
        cmaplist[-1] = 'purple'
        cmaplist[-2] = 'blue'
        cmaplist[-3] = 'navy'
        cmaplist[-4] = 'gold'
        cmaplist[-5] = 'mediumvioletred'
        cmaplist[-6] = 'deepskyblue'
        cmaplist[-7] = 'indigo'
    
    cmap, norm = colors.from_levels_and_colors(unique_labels, cmaplist, extend='max')
    
    return cmap, norm
        
####################################################################################################

def plot_IOM_subspaces_3D(df, minsig=3, savepath=None):
    '''
    Plots the clusters for each combination of clustering features
    
    Parameters:
    df(vaex.DataFrame): Data frame containing the labelled sources.
    minsig(float): Minimum statistical significance level we want to consider.
    savepath(str): Path of the figure to be saved, if None the figure is not saved.
    '''
    
    x_axis = ['Lz/10e2', 'Lz/10e2', 'Lperp/10e2']
    y_axis = ['En/10e3', 'Lperp/10e2', 'En/10e3']
    
    xlabels = ['$L_z$ [$10^3$ kpc km/s]', '$L_z$ [$10^3$ kpc km/s]', '$L_{\perp}$ [$10^3$ kpc km/s]']
    ylabels = ['$E$ [$10^4$ km$^2$/s$^2$]', '$L_{\perp}$ [$10^3$ kpc km/s]', '$E$ [$10^4$ kpc km/s]']
    
    limits =  [[[-4.5, 4.6], [-17.3, 0.7]],
               [[-4.5, 4.6], [-0.03, 3.55]],
               [[-0.03, 3.55], [-17.3, 0.7]]]

    fig, axs = plt.subplots(1,3, figsize = [13, 4])
    
    df_minsig = df[(df.labels>0) & (df.significance>minsig)]
    
    cmap, norm = get_cmap(df_minsig)
   
    for i in range(3):
        plt.sca(axs[i])
        df.scatter(x_axis[i], y_axis[i], s=0.5, c='lightgrey', alpha=0.1, length_limit=60000)
        df_minsig.scatter(x_axis[i], y_axis[i], s=2, c=df_minsig.labels.values,
                                cmap=cmap, norm=norm, alpha=0.6)
        
        plt.xlim(limits[i][0])
        plt.ylim(limits[i][1])
        plt.xlabel(xlabels[i])
        plt.ylabel(ylabels[i])

    plt.tight_layout(w_pad=1)
    
    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

####################################################################################################        

def plot_v_subspaces(df, minsig=3, savepath=None):
    ''''
    Plots the IOM-clusters in velocity space.
    
    Parameters:
    df(vaex.DataFrame): The halo set and its associated labels
    savepath(str): Location where the plot should be saved, if None the plot is not saved
    '''
    x_axis = ['vR', 'vR', 'vz']
    y_axis = ['vphi', 'vz', 'vphi']
    
    xlabels = ['$v_R$ [km/s]', '$v_R$ [km/s]', '$v_{z}$ [km/s]']
    ylabels = ['$v_{\phi}$ [km/s]', '$v_z$ [km/s]', '$v_{\phi}$ [km/s]']
    
    fig, axs = plt.subplots(1,3, figsize = [14, 4]) #figsize = [20,9]
    
    unique_labels = np.unique(df.labels.values)
    
    df_minsig = df[(df.labels>0) & (df.significance>minsig)]
    cmap, norm = get_cmap(df_minsig)
    
    background = df[(df.vR>-500) & (df.vR<500) & (df.vphi>-500) & (df.vphi<500)]
    
    for i in range(3):
        plt.sca(axs[i])
        plt.xlim([-570, 570])
        plt.ylim([-570, 570])
        background.scatter(x_axis[i], y_axis[i], s=0.5, c='lightgrey', alpha=0.1, length_limit=60000)
        df_minsig.scatter(x_axis[i], y_axis[i], s=2, c=df_minsig.labels.values,
                                cmap=cmap, norm=norm, alpha=0.8)
        
        plt.xlabel(xlabels[i])
        plt.ylabel(ylabels[i])
    
    plt.tight_layout(w_pad=1)
    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

######################################################################################################        
        
def plot_original_data_3D(df, savepath=None):
    
    c_args = dict(colormap = 'afmhot', vmin = 0, vmax = 5)

    x_axis = ['Lz/10e2', 'Lz/10e2', 'Lperp/10e2']
    y_axis = ['En/10e3', 'Lperp/10e2', 'En/10e3']
    
    xlabels = ['$L_z$ [$10^3$ kpc km/s]', '$L_z$ [$10^3$ kpc km/s]', '$L_{\perp}$ [$10^3$ kpc km/s]']
    ylabels = ['$E$ [$10^4$ km$^2$/s$^2$]', '$L_{\perp}$ [$10^3$ kpc km/s]', '$E$ [$10^4$ kpc km/s]']
    
    limits =  [[[-4.5, 4.6], [-17.3, 0.7]],
               [[-4.5, 4.6], [-0.02, 4.3]],
               [[-0.02, 4.3], [-17.3, 0.7]]]

    fig, axs = plt.subplots(1,3, figsize = [11.5, 3.5])

    for i in range(3):
        plt.sca(axs[i])
        im = df.plot(x_axis[i], y_axis[i], f='log', colorbar=False, limits=limits[i], **c_args)
        plt.xlabel(xlabels[i])
        plt.ylabel(ylabels[i])
    
    fig.subplots_adjust(right=0.8)
    cb = fig.add_axes([1.01, 0.22, 0.015, 0.73])     
    fig.colorbar(sm(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap='afmhot'), cax=cb, label = "log count (*)")
    
    plt.tight_layout()
    
    if(savepath is not None):
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    else:
        plt.show()    

########################################################################################################

def plot_subgroup_vspace_En_Lz(df, clusterNr, savepath=None):
    '''
    Plots the subgroups in velocity space of a specific cluster.
    
    Parameters:
    df(vaex.DataFrame): Dataframe containing stars and cluster labels
    clusterNr(int): Label of cluster that we want to plot
    savepath(str): Location where the plot should be saved, if None the plot is not saved
    '''
    
    fig, axs = plt.subplots(1,4, figsize = [20,4.5])

    cluster = df[df.labels==clusterNr]

    cmaplist = ['black', 'ForestGreen', 'DodgerBlue', 'MediumVioletRed', 'MediumBlue', 
                'DarkCyan', 'gold', 'deeppink', 'yellow', 'red', 'lime', 'darkolivegreen', 'purple',
                'plum', 'darkgoldenrod', 'darkgrey', 'crimson', 'navy', 'hotpink', 'orange',
                'cadetblue', 'steelblue',
                'ForestGreen', 'DodgerBlue', 'MediumVioletRed', 'MediumBlue', 
                'DarkCyan', 'gold', 'deeppink', 'yellow', 'red', 'lime', 'darkgrey', 'purple',
                'plum', 'darkgoldenrod', 'darkolivegreen', 'crimson', 'navy', 'hotpink', 'orange',
                'cadetblue', 'steelblue',
                'ForestGreen', 'DodgerBlue', 'MediumVioletRed', 'MediumBlue', 
                'DarkCyan', 'gold', 'deeppink', 'yellow', 'red', 'lime', 'darkgrey', 'purple',
                'plum', 'darkgoldenrod', 'darkolivegreen', 'crimson', 'navy', 'hotpink', 'orange',
                'cadetblue', 'steelblue']

    x_axis = ['vR', 'vR', 'vphi']
    y_axis = ['vphi', 'vz', 'vz']
    xlabels = ['$v_R$ [km/s]', '$v_R$ [km/s]', '$v_{\phi}$ [km/s]']
    ylabels = ['$v_{\phi}$ [km/s]', '$v_z$ [km/s]', '$v_z$ [km/s]']
    
    cluster_colors = [cmaplist[col] for col in cluster[cluster.v_space_labels>0].v_space_labels.values]
    
    for i in range(3):
        plt.sca(axs[i])
        axs[i].set_xlim(-400, 400)
        axs[i].set_ylim(-400, 400)
            
        df.scatter(x_axis[i], y_axis[i], s=0.5, c='lightgrey', alpha=0.1, length_limit=60000)
        cluster[cluster.v_space_labels==0].scatter(x_axis[i], y_axis[i], s=7, color="none",
                                                   edgecolor="black", linewidth=0.5, alpha=0.5)
        cluster[cluster.v_space_labels>0].scatter(x_axis[i], y_axis[i], s=7, c=cluster_colors, alpha=0.7)
        plt.xlabel(xlabels[i])
        plt.ylabel(ylabels[i])
    
    plt.sca(axs[3])
    axs[3].set_xlim(-3.2, 3.2)
    axs[3].set_ylim(-1.7, -0.6)
        
    df.scatter('Lz/10e2', 'En/10e4', s=0.5, c='lightgrey', alpha=0.1, length_limit=60000)
    cluster[cluster.v_space_labels==0].scatter('Lz/10e2', 'En/10e4', s=7, color="none",
                                               edgecolor="black", linewidth=0.5, alpha=0.5)
    cluster[cluster.v_space_labels>0].scatter('Lz/10e2', 'En/10e4', s=7, c=cluster_colors,
                                              alpha=0.7, label=f'{clusterNr}')
    
    plt.xlabel('$L_z$ [$10^3$ kpc km/s]')
    plt.ylabel('$E$ [$10^5$ km$^2$/s$^2$]')

    axs[3].text(2.5, -0.7, f'{clusterNr}', 
    weight='bold', horizontalalignment='center', 
    verticalalignment='center',
    fontsize=16, color='black')
    
    plt.tight_layout()
    
    if(savepath is not None):
        plt.savefig(savepath, dpi=100)
    else:
        plt.show()
        
########################################################################################################