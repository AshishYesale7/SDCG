#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           SDCG FALSIFICATION TEST: Void vs Cluster Dwarf Galaxies            ║
║                                                                              ║
║  MASTER RUNNER: Executes all 4 phases in sequence                           ║
║                                                                              ║
║  Usage:                                                                      ║
║    python run_all.py                                                         ║
║                                                                              ║
║  Phases:                                                                     ║
║    1. Download ALFALFA α.100 + SDSS Void Catalog                            ║
║    2. Filter dwarf sample (10^7 < M* < 10^9 M_☉)                            ║
║    3. Statistical analysis (Welch's t-test, bootstrap CI)                   ║
║    4. Interpretation & visualization                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def run_phase(script_name, description):
    """Run a single phase script."""
    script_path = SCRIPT_DIR / script_name
    
    print()
    print("=" * 70)
    print(f"PHASE: {description}")
    print(f"Script: {script_name}")
    print("=" * 70)
    print()
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SCRIPT_DIR),
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\n❌ Phase failed: {script_name}")
        return False
    
    print(f"\n✓ Phase complete: {description}")
    return True


def main():
    """Run all phases of the falsification test."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║     ███████╗██████╗  ██████╗ ██████╗     ████████╗███████╗███████╗████████╗ ║
    ║     ██╔════╝██╔══██╗██╔════╝██╔════╝     ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝ ║
    ║     ███████╗██║  ██║██║     ██║  ███╗       ██║   █████╗  ███████╗   ██║    ║
    ║     ╚════██║██║  ██║██║     ██║   ██║       ██║   ██╔══╝  ╚════██║   ██║    ║
    ║     ███████║██████╔╝╚██████╗╚██████╔╝       ██║   ███████╗███████║   ██║    ║
    ║     ╚══════╝╚═════╝  ╚═════╝ ╚═════╝        ╚═╝   ╚══════╝╚══════╝   ╚═╝    ║
    ║                                                                              ║
    ║              VOID vs CLUSTER DWARF GALAXY ROTATION TEST                      ║
    ║                                                                              ║
    ║         Testing: G_eff(ρ) = G_N × [1 + μ × S(ρ)]                            ║
    ║         Where:   S(ρ) = 1/(1 + (ρ/ρ_thresh)²)                               ║
    ║                                                                              ║
    ║         Prediction: void dwarfs should rotate ~0.5 km/s faster              ║
    ║                     (Lyα-constrained SDCG)                                  ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    phases = [
        ("download_catalogs.py", "1. Download ALFALFA + SDSS Void Catalogs"),
        ("filter_dwarf_sample.py", "2. Filter Dwarf Sample by Mass & Environment"),
        ("run_void_cluster_analysis.py", "3. Statistical Analysis"),
        ("interpret_results.py", "4. Interpretation & Visualization"),
    ]
    
    for script, description in phases:
        success = run_phase(script, description)
        if not success:
            print("\n" + "=" * 70)
            print("PIPELINE FAILED")
            print("=" * 70)
            sys.exit(1)
    
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "    ALL PHASES COMPLETE - FALSIFICATION TEST FINISHED    ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("Key outputs:")
    print(f"  • Results:  {SCRIPT_DIR / 'results' / 'analysis_summary.txt'}")
    print(f"  • Figure:   {SCRIPT_DIR / 'plots' / 'void_cluster_test.pdf'}")
    print(f"  • LaTeX:    {SCRIPT_DIR / 'plots' / 'void_cluster_table.tex'}")
    print()


if __name__ == "__main__":
    main()
