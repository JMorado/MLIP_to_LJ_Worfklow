# Preparing simulations

- This directory is based on the [physical properties](https://github.com/openforcefield/ash-sage-rc2/tree/main/01_download-data/physprop/final) workflow from the ash-sage-rc2 repository.  
- The training dataset is defined in [sage-training-set.csv](output/sage-training-set.csv).  
- The Jupyter notebook [01_1_analyse.ipynb](output/simulations/01_1_analyse.ipynb) automatically reads the training dataset and creates the necessary directories and files required to run the simulations in OpenMM.
  - If a density value is provided, it is used to generate the initial simulation box. Otherwise, a default value of 1 is used.
  - Each system contains approximately 500 molecules (some systems may contain 499 molecules), with 10 ns simulations performed at 1 bar and 298.15 K in the NPT ensemble.
  - The simulation script is defined in the [template file](output/simulations/template.py).
- The script [submit_array.sh](output/simulations/mixtures/submit_array.sh) can be used to run the simulations on `comet`.

# README from the original repository

## Filtering training and validation sets

This directory contains scripts for filtering training and validation sets,
profiling the output datasets, and making plots of the said datasets.

Please see `run.sh` for examples of how to run the scripts, and input/output options.
Please see log files in `logs/` for logging output.


### Training set

This comprised densities and enthalpies of mixing of binary mixtures. In total, 594 densities and 561 enthalpies of mixing were curated.

### Validation sets

These were curated in two stages. The first validation set, or `validation-set.csv|json`, contains 618 densities and 436 enthalpies of mixing. The second set, `validation-set-extended.csv|json`, contains 1777 densities and 547 enthalpies of mixing.

In total, the validation sets contain 2395 densities and 983 enthalpies of mixing.