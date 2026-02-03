<!--
─────────────────────────────────────────────────────────────────────────────
SDCG Project — Copyright (c) 2025, Ashish Vasant Yesale (ashishyesale007@gmail.com)
SPDX-License-Identifier: BSD-3-Clause

This file is part of the SDCG (Scale-Dependent Conformal Gravity) Research Project.

─────────────────────────────────────────────────────────────────────────────
BSD 3-Clause License
---------------------

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

By using this software, you agree to be bound by the terms of this license.

─────────────────────────────────────────────────────────────────────────────
Contributor Guidelines:
------------------------
Contributions are welcome under the terms of the Developer Certificate of Origin (DCO).
All contributors must certify that they have the right to submit the code and agree to
release it under the above license terms.

Contributions must:
  - Be original or appropriately attributed
  - Include clear documentation and test cases where applicable
  - Respect the scientific integrity and coding guidelines defined herein
  - Follow reproducibility standards for computational physics

─────────────────────────────────────────────────────────────────────────────
Terms of Use and Disclaimer:
-----------------------------
This software and research is provided "as is", without any express or implied warranty.
In no event shall the authors, contributors, or copyright holders
be held liable for any damages arising from the use of this software.

Scientific conclusions derived from this code should be independently verified.
Use of this software for publications requires proper citation of the original work.

─────────────────────────────────────────────────────────────────────────────
-->

# Contributing to SDCG

First off — thanks for considering contributing to the **Scale-Dependent Conformal Gravity (SDCG)** project!  
We welcome thoughtful collaboration and rigorous scientific inquiry.

---

## 🧠 Project Vision

**SDCG** is a theoretical cosmology research project exploring:

- **Scale-Dependent Conformal Gravity** as an extension to General Relativity
- **Conformal anomaly-driven** modifications to gravitational dynamics
- **Resolution of cosmological tensions** (H₀, S₈) through screening mechanisms
- **MCMC-based parameter estimation** using real observational data
- **Integration with CLASS and Cobaya** for precision cosmology

### Key Parameters

| Parameter | Value       | Description                            |
| --------- | ----------- | -------------------------------------- |
| μ         | 0.47 ± 0.03 | SDCG coupling strength (MCMC best-fit) |
| μ_bare    | 0.48        | QFT one-loop prediction                |
| β₀        | 0.70        | Conformal anomaly coefficient          |
| ρ_thresh  | 200 ρ_crit  | Screening density threshold            |

---

## 👣 How to Contribute

### 1. **Fork the Repository**

```bash
git clone https://github.com/AshishYesale7/SDCG-Framework.git
cd SDCG
```

### 2. **Create a Feature Branch**

```bash
git checkout -b feature/my-contribution
```

### 3. **Set Up Your Environment**

```bash
# Create virtual environment
python -m venv sdcg_env
source sdcg_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# For CLASS integration
cd class_cgc && make clean && make
```

### 4. **Make Your Changes**

- Follow coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 5. **Submit a Pull Request**

```bash
git add .
git commit -m "feat: description of your contribution"
git push origin feature/my-contribution
```

---

## ✅ Core Design Principles

1. **Reproducibility**: All results must be reproducible from code and data
2. **Real Data Only**: No mock/simulated data in production analysis
3. **Physical Consistency**: All modifications must respect fundamental physics
4. **Transparency**: Clear documentation of assumptions and approximations
5. **Cross-Validation**: Results should be validated against multiple datasets

---

## 🛠️ Technology Stack

### 1. 🔧 **Languages & Frameworks**

| Purpose          | Technology                | Reason                                       |
| ---------------- | ------------------------- | -------------------------------------------- |
| MCMC Analysis    | **Python + emcee/Cobaya** | Industry-standard for cosmological inference |
| Boltzmann Solver | **CLASS (C)**             | Modified for SDCG dynamics                   |
| Data Processing  | **NumPy, SciPy, Pandas**  | Efficient numerical computing                |
| Visualization    | **Matplotlib, GetDist**   | Publication-quality plots                    |
| Configuration    | **YAML**                  | Human-readable parameter files               |

### 2. 📊 **Data Sources**

| Dataset     | Type               | Reference                 |
| ----------- | ------------------ | ------------------------- |
| SPARC       | Rotation curves    | Lelli et al. (2016)       |
| ALFALFA     | HI velocity widths | Haynes et al. (2018)      |
| Local Group | dSph dispersions   | McConnachie (2012)        |
| Planck 2018 | CMB power spectra  | Planck Collaboration      |
| Pantheon+   | Type Ia SNe        | Scolnic et al. (2022)     |
| DESI BAO    | BAO measurements   | DESI Collaboration (2024) |

### 3. 🧪 **Analysis Tools**

| Component                  | Tool                   | Purpose                     |
| -------------------------- | ---------------------- | --------------------------- |
| Parameter Estimation       | **emcee / Cobaya**     | MCMC sampling               |
| Model Comparison           | **GetDist**            | Chain analysis and plotting |
| Cosmological Calculations  | **CLASS-CGC**          | Modified Boltzmann solver   |
| Statistical Analysis       | **SciPy, statsmodels** | Hypothesis testing          |
| Environment Classification | **Custom Python**      | Void/cluster identification |

---

## 📁 Project Structure

```
SDCG/
├── class_cgc/              # Modified CLASS Boltzmann solver
├── data/                   # Observational datasets
│   ├── planck/            # Planck CMB data
│   ├── bao/               # BAO measurements
│   ├── sne/               # Supernovae data
│   └── growth/            # Growth rate data
├── scripts/               # Analysis scripts
├── simulations/           # Pipeline and strategy docs
├── results/               # MCMC chains and outputs
├── plots/                 # Generated figures
└── Run/                   # Main execution scripts
```

---

## 🔬 Scientific Contribution Guidelines

### Types of Contributions Welcome

1. **Theoretical Extensions**
   - New screening mechanisms
   - Alternative parameterizations of μ(ρ)
   - Connections to other modified gravity theories

2. **Data Analysis**
   - New dataset integrations
   - Improved systematic corrections
   - Cross-validation studies

3. **Code Improvements**
   - Performance optimizations
   - Better error handling
   - Enhanced documentation

4. **Documentation**
   - Tutorials and examples
   - Mathematical derivations
   - Installation guides

### Scientific Standards

- All claims must be supported by data or derivation
- Statistical uncertainties must be properly propagated
- Systematic effects must be explicitly discussed
- Prior choices must be justified

---

## 🧑‍💻 Coding Standards

### Python Style

```python
# Use descriptive names
def calculate_sdcg_modification(density, mu=0.47, rho_thresh=200):
    """
    Calculate SDCG velocity modification.

    Parameters
    ----------
    density : float
        Local matter density in units of critical density
    mu : float
        SDCG coupling parameter (default: 0.47)
    rho_thresh : float
        Screening threshold in units of critical density

    Returns
    -------
    float
        Velocity modification in km/s
    """
    if density > rho_thresh:
        return 0.0  # Screened regime
    screening_factor = 1.0 - (density / rho_thresh)
    return mu * screening_factor * VELOCITY_SCALE
```

### Documentation Requirements

- Docstrings for all functions and classes
- Type hints for function signatures
- Comments for complex physics calculations
- README updates for new features

### Testing

```bash
# Run tests before submitting
pytest tests/
python -m pytest --cov=sdcg tests/
```

---

## 🔐 Data Integrity & Reproducibility

### Version Control for Results

- All MCMC chains should be versioned with timestamps
- Random seeds must be recorded for reproducibility
- Configuration files must accompany all runs

### Data Provenance

- Original data sources must be cited
- Any preprocessing steps must be documented
- Derived quantities must include error propagation

---

## 📜 Contributor License Agreement

![CLA](https://img.shields.io/badge/Contributor_License_Agreement-Required-blue.svg)

By contributing to this repository, you grant Ashish Yesale (the project maintainer) an irrevocable, worldwide, royalty-free license to use, modify, and sublicense your contribution under the terms of the SDCG open source license (BSD 3-Clause).

**You retain copyright on your original work.**

You confirm that:

- Your contribution is your own work or you have permission to submit it
- You agree to the terms of the [Code of Conduct](./CODE_OF_CONDUCT.md)
- Your contribution follows scientific integrity standards
- Any data used is properly licensed and attributed

> "Contributors retain copyright to their original work.
> By submitting, you agree to our licensing terms and scientific integrity standards."

---

## 📧 Contact & Questions

| Type                  | Contact                   |
| --------------------- | ------------------------- |
| **General Inquiries** | ashishyesale007@gmail.com |
| **Bug Reports**       | Open a GitHub Issue       |
| **Feature Requests**  | Open a GitHub Discussion  |
| **Security Issues**   | Email maintainer directly |

---

## 🏆 Recognition

All significant contributors will be:

- Listed in the AUTHORS file
- Acknowledged in publications using this code
- Credited in release notes

---

## 📚 Recommended Reading

Before contributing, familiarize yourself with:

1. **SDCG Theory**
   - Conformal gravity fundamentals
   - Screening mechanisms in modified gravity
   - Cosmological tensions (H₀, S₈)

2. **Technical Background**
   - MCMC methods (emcee, Cobaya)
   - Boltzmann solvers (CLASS, CAMB)
   - Bayesian parameter estimation

3. **Key References**
   - Mannheim & Kazanas (1989) — Conformal gravity
   - Lelli et al. (2016) — SPARC database
   - Planck Collaboration (2020) — Planck 2018 results

---

## 🌟 Getting Started — First Contribution

1. **Good First Issues**: Look for issues labeled `good-first-issue`
2. **Documentation**: Help improve docs and tutorials
3. **Testing**: Add test cases for existing code
4. **Data Validation**: Help verify data processing pipelines

---

**Thank you for contributing to advancing our understanding of gravity and cosmology!** 🌌

---

_Last updated: February 2026_
