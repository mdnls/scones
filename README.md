# SCONES: Large-scale Experiments

**Score-based Generative Neural Networks for Large-Scale Optimal Transport**. ([on ArXiv](https://arxiv.org/abs/2110.03237)) <br />
_Max Daniels, Tyler Maunu, Paul Hand_.

This repository contains code for running large scale experiments, such as those used to generate Figure 1 (CelebA sampling). A sister repository, used to run all synthetic/small-scale experiments, can be found [here](https://github.com/mdnls/scones-synthetic).

## Setup
The required packages can be found in `requirements.txt`. To create a new conda environment with these packages installed, use

`conda create --name <env> --file requirements.txt`.

## Running the code
There are four main entry points, each dedicated to training and evaluating four different 'types' of models.
1. `cpat.py` -- Training compatibility functions.
2. `ncsn.py` -- Training score based models for unconditional sampling.
3. `scones.py` -- Conditional sampling SCONES by combining a score-based (unconditional) model with a compatibility function.
4. `bproj.py` -- Training barycentric projection models given a compatibility function.

The general template for using these files is

`python <file>.py --doc <name-for-output> --config <cnf>.yml`

The name `<name-for-output>` will be used as a label for experimental artifacts saved to disk. The config file `<cnf>.yml` must be located in the `config` subdirectory of the top level folder corresponding to `<file>.py`.

For example, to train a compatibility function for transportation from blurry CelebA to CelebA, one could use

`python cpat.py --doc celeba-superres --config superres_KL_0.005.yml`

This corresponds to the configuration file located in `compatibility/configs/superres_KL_0.005.yml`, where one can customize the experiment or view its fine-grained details.
