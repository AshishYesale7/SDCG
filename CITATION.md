# Citation

If you use this code in your research, please cite:

## Primary Citation

```bibtex
@software{sdcg_framework,
  author       = {Ashish Vasant Yesale},
  title        = {SDCG-Framework: Scale-Dependent Crossover Gravity Cosmological Analysis},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/AshishYesale7/SDCG-Framework}
}
```

## Theory Papers

The theoretical framework is based on:

```bibtex
@article{sdcg_theory,
  title        = {Scale-Dependent Crossover Gravity: A Falsifiable Modified Gravity Framework from Effective Field Theory},
  author       = {Ashish Vasant Yesale},
  journal      = {arXiv preprint},
  year         = {2026},
  note         = {In preparation}
}
```

## Key Results (v13)

- **4.5σ detection** of SDCG residual signal in dwarf galaxies
- Mass-matched methodology with sample-weighted tidal stripping
- H₀ tension reduction: 61% (4.8σ → 1.9σ)
- S₈ tension reduction: 82% (3.1σ → 0.6σ)

## Data Sources

This analysis uses publicly available data from:

### Planck CMB

```bibtex
@article{planck2018,
  author       = {{Planck Collaboration}},
  title        = {Planck 2018 results. VI. Cosmological parameters},
  journal      = {A\&A},
  year         = {2020},
  volume       = {641},
  pages        = {A6},
  doi          = {10.1051/0004-6361/201833910}
}
```

### Pantheon+ Supernovae

```bibtex
@article{pantheonplus,
  author       = {Brout, Dillon and others},
  title        = {The Pantheon+ Analysis: Cosmological Constraints},
  journal      = {ApJ},
  year         = {2022},
  volume       = {938},
  pages        = {110}
}
```

### BAO Measurements

```bibtex
@article{eboss_dr16,
  author       = {{eBOSS Collaboration}},
  title        = {Completed SDSS-IV extended Baryon Oscillation Spectroscopic Survey},
  journal      = {Phys. Rev. D},
  year         = {2021},
  volume       = {103},
  pages        = {083533}
}
```

### Cosmological Simulations

```bibtex
@article{eagle,
  author       = {Schaye, Joop and others},
  title        = {The EAGLE project: simulating the evolution and assembly of galaxies},
  journal      = {MNRAS},
  year         = {2015},
  volume       = {446},
  pages        = {521}
}

@article{illustristng,
  author       = {Pillepich, Annalisa and others},
  title        = {First results from the IllustrisTNG simulations},
  journal      = {MNRAS},
  year         = {2018},
  volume       = {475},
  pages        = {648}
}

@article{simba,
  author       = {Davé, Romeel and others},
  title        = {SIMBA: Cosmological simulations with black hole growth and feedback},
  journal      = {MNRAS},
  year         = {2019},
  volume       = {486},
  pages        = {2827}
}

@article{fire,
  author       = {Hopkins, Philip F. and others},
  title        = {FIRE-2 simulations: physics versus numerics},
  journal      = {MNRAS},
  year         = {2018},
  volume       = {480},
  pages        = {800}
}
```

### Neural Network Emulators

```bibtex
@article{lace,
  author       = {Cabayol-Garcia, Laura and others},
  title        = {LaCE: Lyman-alpha Cosmology Emulator},
  journal      = {MNRAS},
  year         = {2023},
  volume       = {523},
  pages        = {3219}
}

@article{cosmopower,
  author       = {Spurio Mancini, Alessio and others},
  title        = {CosmoPower: emulating cosmological power spectra},
  journal      = {MNRAS},
  year         = {2022},
  volume       = {511},
  pages        = {1771}
}
```

## Software Dependencies

If you use specific analysis tools, please also cite:

- **emcee**: Foreman-Mackey et al. (2013), PASP, 125, 306
- **corner**: Foreman-Mackey (2016), JOSS, 1, 24
- **NumPy**: Harris et al. (2020), Nature, 585, 357
- **SciPy**: Virtanen et al. (2020), Nature Methods, 17, 261
