# Transfer and Interference 

Code and analysis for Holton, E., Braun, L., Thompson, J. A. F., Grohn, J., & Summerfield, C. (2025, February 24). Humans and neural networks show similar patterns of transfer and interference during continual learning. https://doi.org/10.31234/osf.io/98ksw_v1


## Project Overview
This repository contains code for analyzing transfer and interference patterns in human participants and artificial neural networks (ANNs) trained on the same schedules. 


## Project Structure

```
transfer-interference/
├── data/
│   ├── participants/
│   │   ├── raw/
│   │   ├── trial_df.csv
│   │   └── human_vonmises_fits.csv
│   └── simulations/
├── figures/
│   ├── figure2_transfer_interference/
│   ├── figure3_anns/
│   └── figure4_individual_differences/
├── notebooks/
│   ├── figure2_transfer_interference.ipynb
│   ├── figure3_anns.ipynb
│   └── figure4_individual_differences.ipynb
├── scripts/
│   ├── 01_preprocess_data.py
│   ├── 02_run_simulations.py
│   └── 03_fit_vonmises.py
└── src/
    ├── analysis/
    ├── models/
    └── utils/
```


## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/eleanorholton/transfer-interference.git
cd transfer-interference


2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  
```

3. Install dependencies:
```bash
# Install packages from requirements.txt
pip install -r requirements.txt

# Install the project in editable mode
pip install -e .
```

Typical installation time: 5-10 minutes on a standard desktop computer

## Analysis Pipeline

1. Preprocess participant data (`scripts/01_preprocess_data.py`)
2. Run neural network simulations (`scripts/02_run_simulations.py`)
3. Fit von Mises mixture models (`scripts/03_fit_vonmises.py`)
4. Generate figures using analysis notebooks in `notebooks/`


## Usage


### Data Preprocessing
```bash
python scripts/01_preprocess_data.py
```

### Running Simulations
```bash
python scripts/02_run_simulations.py rich_50 # simulation ID as found in src/models/ann_experiments
```
### Model fitting
```bash
python scripts/03_fit_vonmises.py participants
python scripts/03_fit_vonmises.py simulations --sim-name rich_50 
```

### Analysis
The main results and plots can be run through the .ipynb notebooks in the `notebooks/` directory:
- `figure2_transfer_interference.ipynb`: Analysis of transfer and interference effects
- `figure3_anns.ipynb`: Neural network simulations
- `figure4_individual_differences.ipynb`: Individual differences analysis

Total reproduction time of all simulations and model fitting: ~2-3 hours on a standard desktop computer



## Data Structure

### Participant Data
- Raw data files in `data/participants/raw/`
- Naming convention: `study[1/2]_[same/near/far]_sub[N].csv`
- Processed trial data in `trial_df.csv`
- Von Mises fits in `human_vonmises_fits.csv`

### Simulation Data
- Organized by simulation type (e.g. rich_50 is rich regime of gamma=0.001, 50 hidden units)
- Each directory contains:
  - Individual simulation results (`.npz` files)
  - Settings file (`settings.json`)
  - Von Mises fits (`*_vonmises_fits.csv`)
  - Parameters for simulation settings stored in src/models/ann_experiments

## Contact

eleanor.holton@psy.ox.ac.uk