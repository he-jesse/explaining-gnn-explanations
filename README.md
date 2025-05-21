# Explaining GNN Explanations With Edge Gradients

This repository contains code to reproduce the experiments in the KDD 2025 paper "Explaining GNN Explanations With Edge Gradients", including the Occlusion, Layerwise Occlusion, and Layerwise Gradient methods used for comparisons.

Requirements:
```
python >= 3.10
pyg >= 2.5
pytorch >= 2.1
captum >= 0.2.0
matplotlib
dgl
```

Note that several of the experiments, particularly generating explanations on the larger node classification datasets, may take quite some time.

## Negative Evidence
The negative evidence experiments are contained in their entirety in `nbks/neg_evidence.ipynb` and `more_multiclass_neg_evidence.ipynb`. Simply run the notebooks.

## Infection
Generating the data, training the network and generating explanations can be done by running `infection.py` in the nbks folder. You will need to provide a directory where results can be saved.
`python nbks/infection.py --directory <path/to/results> --device <device>`
The notebook `nbks/infection.ipynb` can be used to reproduce the plots in the paper, as well as the path search comparison with AMP-ave.
*Make sure you change the directory to the directory where you have results saved in the plotting notebook.*

## Real Data

### Node Classification
To run the node classification experiments (Cora, Citeseer, Cornell, Texas, and Wisconsin), use the shell script `node_classification.sh`. You will need to provide the directory where you would like the datasets to be downloaded, as well as the directory where you would like to save the results.

Run `sh node_classification.sh <dataset> <path/to/download/data> <path/to/results> <device>`

In the interest of efficiency, we provide an alternate script for Occlusion and Layerwise Occlusion explanations in `explain/occlusion_alternate.py`. Instead of calling the Occlusion explainers (which performs |E| forward passes for each vertex), the alternate script computes results for all test vertices in $|E|$ forward passes. To use it, run

`python occlusion_alternate.py --dataset <dataset> --root <path/to/download/data> --directory <path/to/results> --device <device> --architecture <architecture> --layers <layers>`

where `architecture` is one of GCN, GAT, or GIN.

### Molecular Datasets
To run the first two graph classification experiments (MUTAG and PROTEINS), use the shell script `graph_classification.sh`. You will need to provide the directory where you would like the datasets to be downloaded, as well as the directory where you would like to save the results.

Run `sh graph_classification.sh <dataset> <path/to/download/data> <path/to/results> <device>`

### Graph-SST2
The Graph-SST2 graph classification experiments are contained in a separate set of python files.

Use
`sh run_sst2.sh <path/to/download/data> <path/to/results> <device>`

### Plotting
Once all the experiments have been run, the plots can be reproduced in `nbks/real_data_plots.ipynb`.
*Make sure you change the directory to the directory where you have results saved in the plotting notebook.*

## Additional Experiments
This repository also contains an experiment using Simple Graph Convolutions (SGC), a type of linear GCN.

`sh run_sgc.sh <dataset> <path/to/download/data> <path/to/results> <device>`

Similar to the node classification scripts, we also provide an alternative method to compute the Occlusion and Layerwise Occlusion results. 