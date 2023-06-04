# Deep Learning for Computer Vision

## Repo structure
This repo takes advantage of two frameworks: (1) Hydra for configs management and (2) Pytorch Lightning for improving our lives when colaborating and running experiments on different hardware. 

The particular approach of this repo is heavily inspired by [https://youtu.be/w10WrRA-6uI].

## Getting started 
*On HPC:*
1. Create a virtual environment the way you are used to (conda, venv, pyenv, whatever). 
The intended approach is to use `venv`. If `conda` or other virtual environment frameworks are used, the scripts in `/bash/` folder cannot be used but are easily modified. (1 line modification). 

It's very important that you load a new-ish python version before running the getting started script. Do this by: 
```
module load python3/3.10
module load cuda/11.7
```

2. Set environment variables (!!):
```{bash}
export PATH_TO_VENV=#path to your venv
```

3. Run the bash script from the root folder:
```{bash}
./bash/setup.sh
```

This will:
1. Load the HPC modules mentioned above
2. Activate the virtual environment
3. Install the package in editable mode with all requirements

You should further install the requirements for developing (writing code) of the package:
```{bash}
pip install -r requirements-dev.txt
```

## Contribution guide

This repo has protection on the ``main`` branch. Therefore any contribution has to go through a Pull Request. 
