# kernel-regression-dim-reduction
Kernel choice, regression methods, and dimensionality reduction project.

# Running the Project


## Recommended Setup (Conda)

Using Conda ensures that all dependencies are installed with compatible versions.

### 1. Create the environment
From the project root directory:

```bash
conda create -n kernel-regression python=3.10
```

### 2. activate the environemnt
```bash
conda activate kernel-regression
```

### 3. install dependencies

## Notes on Reproducibility

The project assumes the Superconductor dataset (train.csv) is placed in
the data/ directory.

All results reported in the paper can be reproduced by running the notebooks
in order.

Random seeds (if used) are set explicitly inside the modeling notebooks.