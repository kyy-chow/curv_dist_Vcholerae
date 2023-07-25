# Calculating the Curvedness of and Distance Between Surface Meshes

Author: Katherine Chow | y.y.chow@umail.leidenuniv.nl

This repository contains three Python scripts for importing/pre-processing, visualising, and performing statistical analyses on polar _Vibrio cholerae_ inner membrane surface meshes outputted by the [Surface Morphometrics package](https://github.com/GrotjahnLab/surface_morphometrics). The three scripts are:

1. `data_preprocessing.py`: This script imports meshes from VTP files and creates a dataframe that contains all information for all poles. 
2. `data_visualisation.py`: This script plots meshes and creates GIFs with curvedness as the heatmap, and generates other figures used in the report. Additionally, it calculates distances between aligned meshes and creates GIFs and a 3D plot with distance as the heatmap.
3. `statistical_analysis.py`: This script performs _t_-tests.
