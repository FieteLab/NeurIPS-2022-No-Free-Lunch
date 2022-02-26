# MEC/HPC Models Investigation

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

## Contact

Questions? Comments? Interested in collaborating? Open an issue or 
email Rylan Schaeffer (rylanschaeffer@gmail.com).
