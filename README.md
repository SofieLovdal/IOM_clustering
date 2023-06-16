## Clustering in integrals-of-motion space

## General info
This clustering algorithm builds a minimum spanning tree (MST) based on the single-linkage algorithm and proceeds to maximize the statistical significance of the data space by evaluating the density of each connected component in the MST compared to a smooth background, using Poisson statistics and an ellipsoidal cluster boundary. The algorithm extracts all statistically significanct clusters of the data space, and labels noise as data points not belonging to any significant cluster, after that the clusters have been selected at the point of maximum statistical significance in the hierarchical tree. The algorithm has been developed for the astronomy application "Substructure in the stellar halo near the Sun. I. Data-driven clustering in Integrals of Motion space.", but can be applied in an any context where it is meaningful to evaluate a candidate cluster based on a smooth background.

If you use this code, please cite

```
@article{lovdal2022substructure,
  title={Substructure in the stellar halo near the Sun-I. Data-driven clustering in integrals-of-motion space},
  author={L{\"o}vdal, S Sofie and Ruiz-Lara, Tom{\'a}s and Koppelman, Helmer H and Matsuno, Tadafumi and Dodd, Emma and Helmi, Amina},
  journal={Astronomy \& Astrophysics},
  volume={665},
  pages={A57},
  year={2022},
  publisher={EDP Sciences}
}
```

More information can also be found in this paper.
	
## Setup
The dependency Vaex 4.16 requires version 3.9 or 3.10 of Python. 
A minimum working example is runnable with Jupyter notebook by cloning this repository. 
You can install the necessary dependencies with pip install -r requirements.txt 

## Parameters
The parameters used in Lovdal et al. (2022) are

```
N_datasets = 100
linkage_method = 'single'
N_sigma_significance = 3
features = ['scaled_En', 'scaled_Lperp', 'scaled_Lz']
features_to_be_scaled = ['En', 'Lperp', 'Lz']
minmax_values = vaex.from_arrays(En=[-170000, 0], Lperp=[0, 4300], Lz=[-4500,4600])
N_sigma_ellipse_axis = 2.83
min_members = 10
max_members = None
```

Set the catalogue_path and result_path to point to your input data catalogue and results folder, respectively.
The value of N_sigma_ellipse_axis corresponds to the length (in standard deviations) of ellipsoid axes covering a 95.4% extent in n-dimensional space, where n is the dimensionality of your data (n=3 in this case).
