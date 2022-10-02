# No Free Lunch from Deep Learning in Neuroscience

Code corresponding to the NeurIPS 2022 paper "No Free Lunch from Deep Learning in Neuroscience: A Case Study through Models of the Entorhinal-Hippocampal Circuit"

## Setup

### Virtual Environment

We recommend creating a virtual environment. One way to do this is by running the 
following sequence of commands:

`python3 -m venv mec_hpc_venv`

Then activate the virtual environment:

`source mec_hpc_venv/bin/activate`

Ensure pip is up-to-date:

`pip install --upgrade pip`

Then install the required packages:

`pip install -r requirements.txt`

### Repositories

To obtain the relevant repositories, see [zips/README.md](zips/README.md). You can either:
 
1. Manually download zips and place them inside `MEC-HPC-Models-Investigation/zips`, then extract
  them to `mec_hpc_investigations/subprojects`.
   
2. Run `python mec_hpc_investigations/utils/data_setup.py` from the main repository directory,
  which will download and unzip the files.

## Running

Code to run our investigations is located inside our python package `mec_hpc_investigations`.

## Attribution

This repository was forked from [Nayebi et al. 2021's](https://proceedings.neurips.cc/paper/2021/hash/656f0dbf9392657eed7feefc486781fb-Abstract.html)
commit `dfaa3bc03eba9df8aace1541c0724482fbcab75e` on 2022/03/18.

## Contact

Questions? Comments? Interested in collaborating? Open an issue or 
email Rylan Schaeffer (rylanschaeffer@gmail.com) and cc Ila Fiete (fiete@mit.edu).
