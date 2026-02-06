#!/usr/bin/env python3
"""
Detailed Audit Report Generator
================================
Generates specific recommendations based on audit findings.
"""

import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Based on audit findings

def generate_report():
    report = {
        'generated': datetime.now().isoformat(),
        'project': str(PROJECT_ROOT),
        
        'critical_issues': [
            {
                'type': 'Parameter Inconsistency',
                'severity': 'HIGH',
                'description': 'H0 has multiple values across files',
                'files_affected': 29,
                'v12_expected': 67.4,
                'values_found': [70.0, 73.04, 65.1, 69.49, 72.77, 74.61, 61.71],
                'recommendation': 'Standardize to Planck 2018 value: H0 = 67.4 km/s/Mpc'
            },
            {
                'type': 'Result Inconsistency',
                'severity': 'HIGH', 
                'description': 'Multiple result files with different H0 values',
                'files_affected': 33,
                'recommendation': 'Review which result file represents the correct analysis'
            },
            {
                'type': 'JSON Config Mismatch',
                'severity': 'MEDIUM',
                'description': 'Tully-Fisher slope/intercept differ between sdcg_real_data_results.json and sdcg_complete_analysis.json',
                'recommendation': 'Determine which file has correct values'
            }
        ],
        
        'result_file_summary': {
            'h0_range': 'H0 values range from 61.71 to 76.33 km/s/Mpc across result files',
            'files_that_matter': [
                {'file': 'cgc_dwarf_mcmc_results.npz', 'date': '2026-02-06', 'status': 'LATEST', 'summary': 'delta_v = -1.85 ± 2.92 km/s'},
                {'file': 'cgc_lace_comprehensive_v6.npz', 'date': '2026-02-06', 'status': 'LATEST', 'summary': 'mu = 0.41, n_g = 0.65'},
                {'file': 'sdcg_definitive_analysis.npz', 'date': '2026-02-01', 'status': 'KEY', 'summary': 'mu_no_lya = 0.48, tension = 3.9σ'},
            ],
            'potentially_obsolete': [
                'cgc_mcmc_results.npz (H0 = 76.33 - way off from expected)',
                'cgc_results_20260130_*.npz (early runs with varying H0)',
                'cgc_results_20260204_034957.npz (H0 = 65.11, S8 = 0.96 - S8 too high)',
            ]
        },
        
        'json_inconsistencies': {
            'tully_fisher': {
                'sdcg_real_data_results.json': {'slope': 0.243, 'intercept': 1.392},
                'sdcg_complete_analysis.json': {'slope': 0.240, 'intercept': -0.191},
                'recommendation': 'Slope similar, intercept very different - check data sources'
            },
            'lyman_alpha_mu_eff': {
                'sdcg_real_data_results.json': 5.76e-05,
                'sdcg_test_results.json': 5.64e-05,
                'difference_pct': 2.1,
                'status': 'MINOR'
            },
            'mean_velocities': {
                'verified_real_data_test.json': {'void': 47.17, 'cluster': 32.25, 'field': 46.92},
                'expanded_dwarf_dataset.json': {'void': 44.96, 'cluster': 30.31, 'field': 47.06},
                'status': 'DIFFERENT DATASETS - NOT A BUG'
            }
        },
        
        'files_to_update_for_v12': [
            {
                'file': 'fetch_real_cosmology_api.py',
                'issue': 'Contains H0 = 70.0 and 73.04',
                'fix': 'Update to H0 = 67.4 (Planck 2018)'
            },
            {
                'file': 'scripts/_verification/verify_equations.py',
                'issue': 'Contains Omega_m = 2.0 (clearly wrong)',
                'fix': 'Correct to Omega_m = 0.315'
            }
        ],
        
        'data_directory_status': {
            'planck': {'status': 'OK', 'files': 2, 'size_mb': 0.16},
            'bao': {'status': 'OK', 'files': 1, 'size_mb': 0.00},
            'sne': {'status': 'OK', 'files': 5, 'size_mb': 32.30},
            'growth': {'status': 'OK', 'files': 1, 'size_mb': 0.00},
            'lyalpha': {'status': 'OK', 'files': 13, 'size_mb': 0.29},
            'little_things': {'status': 'OK', 'files': 6, 'size_mb': 0.02},
            'cgc_simulations': {'status': 'OK', 'files': 2, 'size_mb': 0.01},
        },
        
        'recommendations': {
            'immediate': [
                '1. Standardize H0 = 67.4 km/s/Mpc across all files',
                '2. Review cgc_dwarf_mcmc_results.npz as the latest definitive dwarf analysis',
                '3. Fix Omega_m in verify_equations.py (currently 2.0, should be 0.315)',
            ],
            'cleanup': [
                '1. Archive old result files (cgc_results_20260130_*)',
                '2. Consolidate duplicate JSON configs',
                '3. Create a single source-of-truth config file for v12 parameters',
            ],
            'documentation': [
                '1. Document which result file is the "official" one for each analysis type',
                '2. Add version tags to result files',
            ]
        }
    }
    
    return report


def print_executive_summary():
    """Print a concise executive summary"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        PROJECT AUDIT - EXECUTIVE SUMMARY                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🔴 CRITICAL ISSUES (3)                                                      ║
║  ────────────────────                                                        ║
║  1. H0 inconsistent: Values range 61.7 - 76.3 km/s/Mpc (v12 expects 67.4)   ║
║  2. Multiple result files with different outcomes                            ║
║  3. Omega_m = 2.0 in verify_equations.py (should be 0.315)                  ║
║                                                                              ║
║  🟡 PARAMETER ISSUES (1243 total)                                            ║
║  ────────────────────────────────                                            ║
║  Most are 'h' parameter false positives in library files (scipy/matplotlib)  ║
║  Key project files needing update:                                           ║
║    • fetch_real_cosmology_api.py (H0 = 70.0/73.04)                          ║
║    • scripts/_verification/verify_equations.py (Omega_m = 2.0)              ║
║                                                                              ║
║  📊 RESULT FILES STATUS                                                      ║
║  ─────────────────────                                                       ║
║  LATEST (Use these):                                                         ║
║    • cgc_dwarf_mcmc_results.npz (2026-02-06) - Δv = -1.85 ± 2.92 km/s       ║
║    • cgc_lace_comprehensive_v6.npz (2026-02-06) - μ = 0.41                  ║
║    • sdcg_definitive_analysis.npz (2026-02-01) - Key Lyα analysis           ║
║                                                                              ║
║  POTENTIALLY OBSOLETE (Archive):                                             ║
║    • cgc_mcmc_results.npz (H0 = 76.33 - incorrect)                          ║
║    • cgc_results_20260130_* (early test runs)                               ║
║    • cgc_results_20260204_034957.npz (S8 = 0.96 - too high)                 ║
║                                                                              ║
║  📁 DATA DIRECTORIES: All OK ✓                                               ║
║  ─────────────────────────────                                               ║
║    planck/ ✓  bao/ ✓  sne/ ✓  growth/ ✓  lyalpha/ ✓  little_things/ ✓      ║
║                                                                              ║
║  📝 PRIORITY ACTIONS                                                         ║
║  ─────────────────                                                           ║
║    1. Create centralized v12_parameters.py as single source of truth        ║
║    2. Update fetch_real_cosmology_api.py with correct H0                    ║
║    3. Fix Omega_m in verify_equations.py                                    ║
║    4. Archive obsolete result files to results/_archive/                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def main():
    # Generate and save detailed report
    report = generate_report()
    
    report_file = PROJECT_ROOT / 'AUDIT_REPORT.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Detailed report saved to: {report_file}")
    
    # Print executive summary
    print_executive_summary()


if __name__ == "__main__":
    main()
