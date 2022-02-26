# MEC/HPC Models Investigation

## Setup

### Virtual Environment

We recommend creating a virtual environment. One way to do this is by running the 
following sequence of commands:

`python3 -m venv mechpc_venv`

Then activate the virtual environment:

`source mechpc_venv/bin/activate`

Ensure pip is up-to-date:

`pip install --upgrade pip`

Then install the required packages:

`pip install -r requirements.txt`

### Repositories

To obtain the relevant repositories, see [zips/README.md](zips/README.md). You can either:
 
1. Manually download zips and place them inside `MEC-HPC-Models-Investigation/zips`, then extract
  them to `investigations/subprojects`.
   
2. Run `python investigations/utils/data_setup.py`, which will download and unzip the files.


## Contact

Questions? Comments? Interested in collaborating? Open an issue or 
email Rylan Schaeffer (rylanschaeffer@gmail.com).
