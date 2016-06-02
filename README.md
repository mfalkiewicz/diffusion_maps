# Diffusion embedding

## Contents

This repository contains some of my diffusion embedding (DE) work.
The docs/ directory contains Jupyter notebooks with diffusion maps tutorial and a simulation demonstrating that DE can successfuly recover ovarlapping gradients of connectivity. The folder also contains plot_surfaces.py function for surface plotting in Pyhton.

The main directory includes diffusion_embed.py file, which contains code for performing DE. 

## Requirements

The code has been written in Python 2.7. The DE code has been integrated into the class object. Following modules are neccessary to run all elements of the code:
- numpy
- scipy
- pySTATIS ( http://github.com/mfalkiewicz/pySTATIS )

## Usage

```
from diffusion_embed import Diffusion_Embedding as de

emb = de(source_path = '/home/me/awesome_project/timeseries/',
                   file_template = '%s_timeseries.npy',
                   subjects = ['22529','23197','23269','23464','23734','23884','24691','24715','24757'],
                   output_path = '/home/me/awesome_project',
                   mwall = False)

emb.compute_embeddings()
emb.realign_embeddings()
emb.project_template_subjects()

```

Following options are available for the Diffusion_Embedding class:

**source_path** (mandatory) - the path where timeseries are stored
**file_template** (mandatory) - format of files to be read in
**subjects** (mandatory) - the changing part of file_template
**output_path** (mandatory) - where to store output files
**diff_time** (optional, default 0 ) - diffusion time
**diff_alpha** (optional, default 0.5) - diffusion operator
**diff_ncomp** (optional, default 10) - number of components to save
**subjects_subset** (optional, default None) - subset of subjects to use for template creation and back projection 
**output_suffix** (optional, default 'embedding') - suffix to add to output filenames 
**ftype** (optional, default 'npy_timeseries') - type of input data
**surf** (optional, default 'fsaverage4') - FreeSurfer template for the data, in calse mwall = True 
**mwall** (optional, default False) - if True, medial wall vertices will be removed from analysis based on surf parameter 
**tp** (optional, default None) - timepoints to extract from timeseries 
**affinity_metric** (optional, default 'correlation') - affinity metric to use between timeseries 
**realign_method** (optional, default 'STATIS') - method of aligning subjects into a common space (basis)
