#!/usr/bin/env python3
"""
Comprehensive Project Audit Tool
================================

Deep survey of MCMC_cgc project to identify:
1. Files not up to date with v12 parameters
2. Duplicate files serving same purpose
3. Parameter inconsistencies
4. Result file analysis
5. Data file health check

Run: python comprehensive_audit.py
"""

import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Known v12 (latest) parameter values for SDCG/CGC model
V12_PARAMETERS = {
    # Cosmological parameters (Planck 2018)
    'H0': 67.4,
    'h': 0.674,
    'Omega_m': 0.315,
    'Omega_b': 0.0493,
    'Omega_cdm': 0.266,
    'Omega_Lambda': 0.685,
    'sigma8': 0.811,
    'n_s': 0.965,
    'tau': 0.054,
    
    # CGC/SDCG specific
    'alpha_cgc': 0.48,
    'beta_cgc': 1.2,
    'gamma_cgc': 0.85,
    'delta_v_strip': 7.9,  # km/s from simulations
    
    # Physical constants
    'c': 299792.458,  # km/s
    'G': 6.674e-11,   # N m^2/kg^2
}

# Old/deprecated parameter values to flag
DEPRECATED_PARAMS = {
    'H0': [70.0, 72.0, 73.0, 67.0],  # Old Hubble values
    'Omega_m': [0.3, 0.27, 0.32],
    'sigma8': [0.8, 0.82, 0.83],
}

class Colors:
    """Terminal colors for output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text, color=Colors.CYAN):
    """Print colored header"""
    print(f"\n{color}{Colors.BOLD}{'='*70}")
    print(f" {text}")
    print(f"{'='*70}{Colors.END}\n")

def print_subheader(text, color=Colors.BLUE):
    """Print colored subheader"""
    print(f"\n{color}{Colors.BOLD}--- {text} ---{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠  {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗  {text}{Colors.END}")

def print_success(text):
    print(f"{Colors.GREEN}✓  {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.WHITE}   {text}{Colors.END}")


class ProjectAuditor:
    def __init__(self, root_path):
        self.root = Path(root_path)
        self.issues = []
        self.warnings = []
        self.duplicates = defaultdict(list)
        self.parameter_files = {}
        self.result_files = []
        self.all_files = []
        self.file_hashes = defaultdict(list)
        
    def collect_files(self):
        """Collect all relevant files"""
        skip_dirs = {'__pycache__', 'cgc_env', '.git', 'node_modules', '.venv', 'venv'}
        skip_patterns = ['lower_bound', 'upper_bound', 'Lower_Bound', 'Upper_Bound']
        
        for root, dirs, files in os.walk(self.root):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                filepath = Path(root) / file
                
                # Skip bound files as requested
                if any(p in str(filepath) for p in skip_patterns):
                    continue
                    
                # Collect relevant files
                if file.endswith(('.py', '.json', '.yaml', '.yml', '.csv', '.txt', '.npz', '.npy')):
                    self.all_files.append(filepath)
                    
    def calculate_file_hash(self, filepath):
        """Calculate MD5 hash of file content"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None
            
    def find_duplicates(self):
        """Find duplicate files by content hash"""
        print_subheader("Finding Duplicate Files (Same Content)")
        
        for filepath in self.all_files:
            if filepath.suffix in ['.py', '.json', '.yaml', '.csv']:
                file_hash = self.calculate_file_hash(filepath)
                if file_hash:
                    self.file_hashes[file_hash].append(filepath)
                    
        duplicates_found = 0
        for hash_val, files in self.file_hashes.items():
            if len(files) > 1:
                duplicates_found += 1
                print_warning(f"Duplicate content found ({len(files)} files):")
                for f in files:
                    print_info(f"  - {f.relative_to(self.root)}")
                    
        if duplicates_found == 0:
            print_success("No exact duplicate files found")
        else:
            print_warning(f"Found {duplicates_found} sets of duplicate files")
            
    def find_similar_purpose_files(self):
        """Find files that might serve similar purposes"""
        print_subheader("Files Potentially Serving Same Purpose")
        
        # Group by base name patterns
        patterns = defaultdict(list)
        for filepath in self.all_files:
            if filepath.suffix == '.py':
                # Extract base pattern (remove numbers, dates, versions)
                name = filepath.stem
                base = re.sub(r'_?\d+|_v\d+|_\d{8}', '', name).lower()
                patterns[base].append(filepath)
                
        similar_found = 0
        for pattern, files in patterns.items():
            if len(files) > 1 and len(pattern) > 3:
                similar_found += 1
                print_warning(f"Similar files ({pattern}):")
                for f in sorted(files):
                    stat = f.stat()
                    print_info(f"  - {f.relative_to(self.root)} ({stat.st_size} bytes, modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d')})")
                    
        if similar_found == 0:
            print_success("No obviously similar-named files found")
            
    def check_parameters_in_file(self, filepath):
        """Check a file for parameter values"""
        issues = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Check for deprecated parameters
            for param, old_values in DEPRECATED_PARAMS.items():
                for old_val in old_values:
                    # Look for assignments like H0 = 70.0 or "H0": 70.0
                    patterns = [
                        rf'{param}\s*=\s*{old_val}(?:\s|$|,)',
                        rf'"{param}"\s*:\s*{old_val}(?:\s|$|,)',
                        rf"'{param}'\s*:\s*{old_val}(?:\s|$|,)",
                    ]
                    for pattern in patterns:
                        if re.search(pattern, content):
                            issues.append({
                                'file': filepath,
                                'param': param,
                                'old_value': old_val,
                                'current_v12': V12_PARAMETERS.get(param, 'N/A')
                            })
                            
            # Check for hardcoded cosmological values that differ from v12
            cosmo_checks = [
                (r'H0\s*=\s*([\d.]+)', 'H0'),
                (r'h\s*=\s*([\d.]+)', 'h'),
                (r'Omega_m\s*=\s*([\d.]+)', 'Omega_m'),
                (r'omega_m\s*=\s*([\d.]+)', 'Omega_m'),
                (r'sigma8\s*=\s*([\d.]+)', 'sigma8'),
                (r'sigma_8\s*=\s*([\d.]+)', 'sigma8'),
            ]
            
            for pattern, param in cosmo_checks:
                matches = re.findall(pattern, content)
                for match in matches:
                    try:
                        val = float(match)
                        expected = V12_PARAMETERS.get(param)
                        if expected and abs(val - expected) > 0.01 * expected:  # More than 1% different
                            issues.append({
                                'file': filepath,
                                'param': param,
                                'old_value': val,
                                'current_v12': expected
                            })
                    except:
                        pass
                        
        except Exception as e:
            pass
            
        return issues
        
    def audit_parameters(self):
        """Audit all files for outdated parameters"""
        print_subheader("Parameter Version Audit (v12 Compliance)")
        
        all_issues = []
        files_checked = 0
        
        for filepath in self.all_files:
            if filepath.suffix in ['.py', '.json', '.yaml']:
                files_checked += 1
                issues = self.check_parameters_in_file(filepath)
                all_issues.extend(issues)
                
        if all_issues:
            # Group by parameter
            by_param = defaultdict(list)
            for issue in all_issues:
                by_param[issue['param']].append(issue)
                
            for param, issues in by_param.items():
                print_error(f"Parameter '{param}' - Found {len(issues)} files with outdated values:")
                print_info(f"  v12 expected: {V12_PARAMETERS.get(param, 'N/A')}")
                for issue in issues[:5]:  # Show first 5
                    print_info(f"  - {issue['file'].relative_to(self.root)}: {issue['old_value']}")
                if len(issues) > 5:
                    print_info(f"  ... and {len(issues) - 5} more files")
        else:
            print_success(f"All {files_checked} files appear to use v12 parameters")
            
        return all_issues
        
    def analyze_result_files(self):
        """Analyze result files for consistency"""
        print_subheader("Result Files Analysis")
        
        results_dir = self.root / 'results'
        if not results_dir.exists():
            print_warning("No results/ directory found")
            return
            
        result_files = list(results_dir.glob('**/*'))
        result_files = [f for f in result_files if f.is_file()]
        
        print_info(f"Found {len(result_files)} result files")
        
        # Group by type
        npz_files = [f for f in result_files if f.suffix == '.npz']
        npy_files = [f for f in result_files if f.suffix == '.npy']
        txt_files = [f for f in result_files if f.suffix == '.txt']
        json_files = [f for f in result_files if f.suffix == '.json']
        
        print_info(f"  .npz: {len(npz_files)}, .npy: {len(npy_files)}, .txt: {len(txt_files)}, .json: {len(json_files)}")
        
        # Analyze NPZ files
        if npz_files:
            print_subheader("NPZ Result Files Comparison")
            import numpy as np
            
            npz_data = {}
            for npz_file in npz_files:
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    npz_data[npz_file.name] = {
                        'keys': list(data.keys()),
                        'mtime': datetime.fromtimestamp(npz_file.stat().st_mtime),
                        'size': npz_file.stat().st_size
                    }
                    # Check for key result values
                    for key in data.keys():
                        arr = data[key]
                        if hasattr(arr, 'shape') and arr.size > 0:
                            if arr.size == 1:
                                npz_data[npz_file.name][f'{key}_value'] = float(arr)
                            elif arr.ndim == 1 and arr.size < 100:
                                npz_data[npz_file.name][f'{key}_mean'] = float(np.mean(arr))
                                npz_data[npz_file.name][f'{key}_std'] = float(np.std(arr))
                except Exception as e:
                    print_warning(f"Could not read {npz_file.name}: {e}")
                    
            for name, info in sorted(npz_data.items(), key=lambda x: x[1]['mtime']):
                print_info(f"\n  {name}:")
                print_info(f"    Modified: {info['mtime'].strftime('%Y-%m-%d %H:%M')}")
                print_info(f"    Keys: {', '.join(info['keys'][:5])}{'...' if len(info['keys']) > 5 else ''}")
                # Show any extracted values
                for k, v in info.items():
                    if k.endswith('_mean') or k.endswith('_value'):
                        print_info(f"    {k}: {v:.4f}")
                        
        # Check for result inconsistencies
        if len(txt_files) > 1:
            print_subheader("Text Summary Files")
            for txt_file in txt_files:
                try:
                    with open(txt_file, 'r') as f:
                        content = f.read()
                    print_info(f"\n  {txt_file.name}:")
                    # Extract key numbers
                    h0_match = re.search(r'H0\s*[=:]\s*([\d.]+)', content)
                    if h0_match:
                        print_info(f"    H0: {h0_match.group(1)}")
                except:
                    pass
                    
    def analyze_json_configs(self):
        """Analyze JSON configuration files for consistency"""
        print_subheader("JSON Configuration Analysis")
        
        json_files = [f for f in self.all_files if f.suffix == '.json']
        
        config_values = defaultdict(dict)
        
        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                    
                if isinstance(data, dict):
                    # Flatten and collect key-value pairs
                    def flatten(d, prefix=''):
                        items = {}
                        for k, v in d.items():
                            key = f"{prefix}.{k}" if prefix else k
                            if isinstance(v, dict):
                                items.update(flatten(v, key))
                            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                                items[key] = v
                        return items
                        
                    flat = flatten(data)
                    for k, v in flat.items():
                        config_values[k][jf.name] = v
            except:
                pass
                
        # Find inconsistencies
        inconsistencies = []
        for key, file_values in config_values.items():
            if len(file_values) > 1:
                unique_values = set(file_values.values())
                if len(unique_values) > 1:
                    inconsistencies.append((key, file_values))
                    
        if inconsistencies:
            print_warning(f"Found {len(inconsistencies)} parameter inconsistencies across JSON files:")
            for key, file_values in inconsistencies[:10]:
                print_error(f"\n  '{key}' has different values:")
                for fname, val in file_values.items():
                    print_info(f"    {fname}: {val}")
        else:
            print_success("No parameter inconsistencies found in JSON files")
            
    def check_data_directories(self):
        """Check data directories for completeness"""
        print_subheader("Data Directory Health Check")
        
        data_dir = self.root / 'data'
        if not data_dir.exists():
            print_error("No data/ directory found!")
            return
            
        # Expected data directories
        expected_dirs = ['planck', 'bao', 'sne', 'growth', 'lyalpha', 'little_things', 'cgc_simulations']
        
        for dirname in expected_dirs:
            subdir = data_dir / dirname
            if subdir.exists():
                n_files = len(list(subdir.glob('**/*')))
                total_size = sum(f.stat().st_size for f in subdir.glob('**/*') if f.is_file())
                size_mb = total_size / (1024 * 1024)
                
                if n_files == 0 or (n_files == 1 and total_size < 100):
                    print_warning(f"  {dirname}/: EMPTY or minimal ({n_files} files)")
                else:
                    print_success(f"  {dirname}/: {n_files} files, {size_mb:.2f} MB")
            else:
                print_error(f"  {dirname}/: MISSING")
                
    def check_imports_consistency(self):
        """Check for import consistency across Python files"""
        print_subheader("Import Consistency Check")
        
        import_counts = defaultdict(int)
        local_imports = defaultdict(list)
        
        for filepath in self.all_files:
            if filepath.suffix == '.py':
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    # Find imports
                    imports = re.findall(r'^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_.]*)', content, re.MULTILINE)
                    for imp in imports:
                        import_counts[imp.split('.')[0]] += 1
                        
                    # Find local imports that might fail
                    local = re.findall(r'^from\s+\.([a-zA-Z_][a-zA-Z0-9_.]*)', content, re.MULTILINE)
                    if local:
                        local_imports[filepath.name] = local
                except:
                    pass
                    
        # Show most common imports
        print_info("Most commonly imported packages:")
        for pkg, count in sorted(import_counts.items(), key=lambda x: -x[1])[:15]:
            print_info(f"  {pkg}: {count} files")
            
        if local_imports:
            print_warning(f"\nFiles with local/relative imports ({len(local_imports)}):")
            for fname, imps in list(local_imports.items())[:10]:
                print_info(f"  {fname}: {', '.join(imps)}")
                
    def analyze_modification_times(self):
        """Analyze file modification times to find outdated files"""
        print_subheader("File Age Analysis")
        
        now = datetime.now()
        age_buckets = {
            'Last 24h': [],
            'Last week': [],
            'Last month': [],
            'Older': []
        }
        
        for filepath in self.all_files:
            if filepath.suffix == '.py':
                try:
                    mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                    age_days = (now - mtime).days
                    
                    if age_days < 1:
                        age_buckets['Last 24h'].append((filepath, mtime))
                    elif age_days < 7:
                        age_buckets['Last week'].append((filepath, mtime))
                    elif age_days < 30:
                        age_buckets['Last month'].append((filepath, mtime))
                    else:
                        age_buckets['Older'].append((filepath, mtime))
                except:
                    pass
                    
        for bucket, files in age_buckets.items():
            n = len(files)
            color = Colors.GREEN if bucket == 'Last 24h' else Colors.YELLOW if bucket == 'Last week' else Colors.WHITE
            print(f"{color}  {bucket}: {n} Python files{Colors.END}")
            
        # Show oldest files (potentially outdated)
        print_warning("\n  Oldest Python files (potentially outdated):")
        all_py = [(f, m) for bucket in age_buckets.values() for f, m in bucket]
        for filepath, mtime in sorted(all_py, key=lambda x: x[1])[:10]:
            print_info(f"    {filepath.relative_to(self.root)}: {mtime.strftime('%Y-%m-%d')}")
            
    def generate_summary(self):
        """Generate final summary"""
        print_header("AUDIT SUMMARY", Colors.MAGENTA)
        
        total_py = len([f for f in self.all_files if f.suffix == '.py'])
        total_json = len([f for f in self.all_files if f.suffix == '.json'])
        total_data = len([f for f in self.all_files if f.suffix in ['.csv', '.npz', '.npy', '.txt']])
        
        print(f"""
{Colors.BOLD}Project Statistics:{Colors.END}
  Total Python files: {total_py}
  Total JSON files: {total_json}
  Total data files: {total_data}
  
{Colors.BOLD}v12 Parameter Reference:{Colors.END}
  H0 = {V12_PARAMETERS['H0']} km/s/Mpc (Planck 2018)
  Omega_m = {V12_PARAMETERS['Omega_m']}
  Omega_b = {V12_PARAMETERS['Omega_b']}
  sigma8 = {V12_PARAMETERS['sigma8']}
  n_s = {V12_PARAMETERS['n_s']}
  
{Colors.BOLD}CGC/SDCG Parameters:{Colors.END}
  alpha_cgc = {V12_PARAMETERS['alpha_cgc']}
  beta_cgc = {V12_PARAMETERS['beta_cgc']}
  delta_v_strip = {V12_PARAMETERS['delta_v_strip']} km/s
""")

    def run_full_audit(self):
        """Run complete audit"""
        print_header("COMPREHENSIVE PROJECT AUDIT", Colors.CYAN)
        print(f"Project: {self.root}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Collect files
        print_info("Collecting files...")
        self.collect_files()
        print_success(f"Found {len(self.all_files)} relevant files")
        
        # Run all checks
        self.find_duplicates()
        self.find_similar_purpose_files()
        param_issues = self.audit_parameters()
        self.analyze_json_configs()
        self.analyze_result_files()
        self.check_data_directories()
        self.check_imports_consistency()
        self.analyze_modification_times()
        
        # Summary
        self.generate_summary()
        
        return {
            'total_files': len(self.all_files),
            'param_issues': len(param_issues),
            'duplicates': sum(1 for v in self.file_hashes.values() if len(v) > 1)
        }


def main():
    auditor = ProjectAuditor(PROJECT_ROOT)
    results = auditor.run_full_audit()
    
    print_header("AUDIT COMPLETE", Colors.GREEN)
    print(f"Total files analyzed: {results['total_files']}")
    print(f"Parameter issues found: {results['param_issues']}")
    print(f"Duplicate file sets: {results['duplicates']}")
    

if __name__ == "__main__":
    main()
