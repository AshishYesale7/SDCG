# Casimir-Gravity Crossover (CGC) Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A phenomenological framework for addressing cosmological tensions through environment-dependent gravitational modifications motivated by vacuum energy physics.

## 🔬 Key Results

| Metric         | ΛCDM | CGC           | Improvement       |
| -------------- | ---- | ------------- | ----------------- |
| Hubble Tension | 4.8σ | 1.9σ          | **61% reduction** |
| S₈ Tension     | 3.1σ | 0.6σ          | **82% reduction** |
| μ (coupling)   | 0    | 0.149 ± 0.025 | **6σ detection**  |

## 📋 Overview

The CGC framework introduces:

- **Environment-dependent gravity**: 14.9% enhancement in low-density voids
- **Built-in screening**: Standard gravity preserved in high-density regions (Solar System safe)
- **Scale-dependent growth**: Testable prediction for DESI Year 5

## 🧪 Two-Front Falsification

### Front 1: CGC Laboratory Test (Immediate)

- Gold plate configuration at d_c ≈ 95 μm
- Tests vacuum-gravity coupling mechanism
- Feasible with current AFM technology

### Front 2: Cosmological Test (2029)

- DESI Year 5 scale-dependent growth measurement
- Predicts f(k=0.1)/f(k=0.01) = 1.10 ± 0.02
- > 5σ discrimination from ΛCDM

## 📁 Repository Structure

```
CGC-Framework/
├── data/                    # Cosmological datasets
│   ├── planck/             # Planck 2018 CMB data
│   ├── bao/                # BOSS DR12 BAO measurements
│   ├── sne/                # Pantheon+ supernovae
│   ├── growth/             # RSD f*sigma8 compilation
│   └── lyalpha/            # eBOSS Lyman-alpha
├── plots/                   # Generated figures
├── results/                 # MCMC chains and analysis outputs
├── class_cgc/              # Modified CLASS cosmology code
├── CGC_THESIS_CHAPTER_v4.tex    # Main thesis document
├── CGC_THESIS_CHAPTER_v4.pdf    # Compiled thesis
├── main_cgc_analysis.py    # Primary MCMC analysis
├── cgc_equations_unified.py # CGC mathematical framework
├── cgc_falsifiability.py   # Falsifiable predictions
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/[username]/CGC-Framework.git
cd CGC-Framework

# Create virtual environment
python3 -m venv cgc_env
source cgc_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run MCMC analysis
python main_cgc_analysis.py
```

## 📊 Datasets Used

| Dataset     | Observable      | Redshift      | Source                                         |
| ----------- | --------------- | ------------- | ---------------------------------------------- |
| Planck 2018 | CMB TT spectrum | z ≈ 1090      | [ESA](https://pla.esac.esa.int)                |
| BOSS DR12   | BAO D_V/r_d     | z = 0.38-0.61 | [SDSS](https://www.sdss.org/dr12/)             |
| Pantheon+   | SNe Ia μ(z)     | z = 0.001-2.3 | [GitHub](https://github.com/PantheonPlusSH0ES) |
| SH0ES 2022  | Local H₀        | z ≈ 0         | [GitHub](https://github.com/PantheonPlusSH0ES) |
| RSD         | f\*σ₈(z)        | z = 0.02-1.48 | Sagredo et al. (2018)                          |

## 📐 Core Equations

**Effective gravitational constant:**

```
G_eff(k,z,ρ)/G_N = 1 + μ · f(k) · g(z) · S(ρ)
```

Where:

- `f(k) = (k/k_pivot)^n_g` — Scale dependence
- `g(z) = exp[-(z-z_trans)²/2σ_z²]` — Redshift window
- `S(ρ) = 1/[1 + (ρ/ρ_thresh)^α]` — Density screening

**CGC Laboratory Test crossover:**

```
d_c = (π ℏc / 480 G σ²)^(1/4)
```

For gold plates (1 μm thick): d_c ≈ 95 μm

## 📚 Citation

If you use this code or framework, please cite:

```bibtex
@article{Yesale2026CGC,
  author = {Yesale, Ashish Vasant},
  title = {Casimir-Gravity Crossover Framework: A Phenomenological Ansatz for Cosmological Tensions},
  year = {2026},
  note = {Available at: https://github.com/[username]/CGC-Framework}
}
```

## 📄 References

1. Planck Collaboration, A&A 641, A6 (2020)
2. Riess et al., ApJL 934, L7 (2022)
3. Scolnic et al., ApJ 938, 113 (2022)
4. Casimir, Proc. Kon. Ned. Akad. Wetensch. B 51, 793 (1948)
5. Lamoreaux, PRL 78, 5 (1997)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Ashish Vasant Yesale**

- Independent Researcher
- Email: ashish.yesale@example.edu

---

_"The CGC framework is offered not as a finished theory, but as a bold hypothesis with concrete predictions. Science advances through such testable proposals—and their honest confrontation with data."_
