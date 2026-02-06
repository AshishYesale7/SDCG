# Installation Guide

## Requirements

- Python 3.10 or later (Python 3.12 recommended)
- pip (Python package manager)

## Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/MCMC_cgc.git
cd MCMC_cgc
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv simulations/cgc_env

# Activate it
# On macOS/Linux:
source simulations/cgc_env/bin/activate

# On Windows:
simulations\cgc_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install numpy scipy matplotlib emcee corner tqdm astropy
```

### 4. Verify Installation

```bash
python -c "
import sys
sys.path.insert(0, '.')
from simulations.cgc.config import PATHS
print('Installation successful!')
print('Base directory:', PATHS['base'])
"
```

## Optional: LaCE Environment

For Lyα forest analysis with LaCE:

```bash
python3 -m venv simulations/lace_env
source simulations/lace_env/bin/activate  # or Windows equivalent
pip install numpy scipy matplotlib h5py configobj
```

## Directory Structure

After installation, your directory should look like:

```
MCMC_cgc/
├── simulations/
│   ├── cgc/           # Core CGC module
│   ├── cgc_env/       # Virtual environment (created by you)
│   ├── class_cgc/     # Modified CLASS Boltzmann solver
│   └── ...
├── data/              # Observational data
├── scripts/           # Analysis scripts
├── results/           # Output directory
└── plots/             # Generated figures
```

## Running Your First Analysis

```bash
# Activate environment
source simulations/cgc_env/bin/activate

# Run a quick test
python -c "
import sys
sys.path.insert(0, '.')
from simulations.cgc.parameters import CGCParameters
params = CGCParameters()
print('CGC Parameters:')
print(f'  μ_fit = {params.mu}')
print(f'  n_g = {params.n_g} (fixed)')
print(f'  z_trans = {params.z_trans} (fixed)')
"
```

## Troubleshooting

### ImportError: No module named 'simulations'

Make sure you're running from the project root directory and have added it to the path:

```python
import sys
sys.path.insert(0, '/path/to/MCMC_cgc')
```

### numpy issues

If you encounter numpy compatibility issues, try:

```bash
pip install numpy==2.0.0
```

## Next Steps

- See [README.md](README.md) for project overview
- See [data/README.md](data/README.md) for data documentation
- See [simulations/README.md](simulations/README.md) for simulation details
