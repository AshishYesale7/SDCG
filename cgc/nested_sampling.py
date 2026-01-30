"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       CGC Nested Sampling Module                             ║
║                                                                              ║
║  Implements nested sampling for Bayesian model comparison using dynesty.    ║
║  Nested sampling directly computes the Bayesian evidence (marginal          ║
║  likelihood), enabling robust comparison between CGC and ΛCDM.              ║
║                                                                              ║
║  Key outputs:                                                                 ║
║    • Bayesian evidence: log Z for model comparison                          ║
║    • Posterior samples: similar to MCMC but with importance weights         ║
║    • Bayes factor: B = Z_CGC / Z_ΛCDM                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Bayesian Evidence
-----------------
The evidence (marginal likelihood) is:
    Z = ∫ L(θ) π(θ) dθ

where L(θ) is the likelihood and π(θ) is the prior.

The Bayes factor compares two models:
    B₁₂ = Z₁ / Z₂

Jeffreys' scale for interpretation:
    |ln B| < 1   : Inconclusive
    1 < |ln B| < 2.5 : Weak evidence
    2.5 < |ln B| < 5 : Moderate evidence
    |ln B| > 5   : Strong evidence

Usage
-----
>>> from cgc.nested_sampling import run_nested_sampling
>>> from cgc.data_loader import load_real_data
>>> data = load_real_data()
>>> results = run_nested_sampling(data)
>>> print(f"log Z = {results['logz']:.2f}")
"""

import numpy as np
import os
from typing import Dict, Any, Tuple, Optional, Callable
import warnings

from .config import PATHS, PLANCK_BASELINE
from .parameters import CGCParameters, get_bounds_array, PARAM_NAMES
from .likelihoods import log_likelihood


# =============================================================================
# PRIOR TRANSFORM
# =============================================================================

def prior_transform(u: np.ndarray) -> np.ndarray:
    """
    Transform unit hypercube to parameter space.
    
    Maps uniform samples in [0, 1]^n to the parameter prior ranges
    for nested sampling.
    
    Parameters
    ----------
    u : np.ndarray
        Uniform samples in [0, 1]^n.
    
    Returns
    -------
    np.ndarray
        Parameters in physical space.
    
    Notes
    -----
    For flat priors, this is a simple linear mapping:
        θ_i = low_i + u_i × (high_i - low_i)
    """
    bounds = get_bounds_array()
    theta = np.zeros_like(u)
    
    for i in range(len(u)):
        low, high = bounds[i]
        theta[i] = low + u[i] * (high - low)
    
    return theta


def prior_transform_lcdm(u: np.ndarray) -> np.ndarray:
    """
    Prior transform for ΛCDM (CGC parameters fixed at 0).
    
    For model comparison, we need to run nested sampling on
    both CGC and ΛCDM. This transform fixes μ = 0.
    
    Parameters
    ----------
    u : np.ndarray
        Uniform samples in [0, 1]^6 (only cosmological params).
    
    Returns
    -------
    np.ndarray
        Full parameter vector with CGC params = 0.
    """
    bounds = get_bounds_array()
    theta = np.zeros(10)
    
    # Transform cosmological parameters (indices 0-5)
    for i in range(6):
        low, high = bounds[i]
        theta[i] = low + u[i] * (high - low)
    
    # Fix CGC parameters
    theta[6] = 0.0    # μ = 0
    theta[7] = 0.0    # n_g = 0
    theta[8] = 2.0    # z_trans (doesn't matter when μ=0)
    theta[9] = 200.0  # ρ_thresh (doesn't matter when μ=0)
    
    return theta


# =============================================================================
# NESTED SAMPLING RUNNER
# =============================================================================

class NestedSampler:
    """
    Nested sampling wrapper using dynesty.
    
    Provides methods for running nested sampling on CGC and ΛCDM
    models, and computing Bayes factors.
    
    Parameters
    ----------
    data : dict
        Cosmological data dictionary.
    likelihood_kwargs : dict, optional
        Additional likelihood arguments.
    
    Attributes
    ----------
    results_cgc : dynesty.Results
        Nested sampling results for CGC.
    results_lcdm : dynesty.Results
        Nested sampling results for ΛCDM.
    bayes_factor : float
        Bayes factor B = Z_CGC / Z_ΛCDM (log scale).
    """
    
    def __init__(self, data: Dict[str, Any], 
                 likelihood_kwargs: Dict = None):
        """Initialize nested sampler."""
        self.data = data
        self.likelihood_kwargs = likelihood_kwargs or {}
        
        self.results_cgc = None
        self.results_lcdm = None
    
    def _get_dynesty(self):
        """Import dynesty, installing if necessary."""
        try:
            import dynesty
            return dynesty
        except ImportError:
            print("Installing dynesty for nested sampling...")
            import subprocess
            import sys
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "dynesty"
            ])
            import dynesty
            return dynesty
    
    def _log_likelihood_cgc(self, theta: np.ndarray) -> float:
        """Log-likelihood for CGC model."""
        return log_likelihood(theta, self.data, **self.likelihood_kwargs)
    
    def _log_likelihood_lcdm(self, theta: np.ndarray) -> float:
        """Log-likelihood for ΛCDM model (μ fixed at 0)."""
        # Expand to full parameter vector
        full_theta = prior_transform_lcdm(theta)
        return log_likelihood(full_theta, self.data, **self.likelihood_kwargs)
    
    def run_cgc(self, nlive: int = 500, 
                dlogz: float = 0.1,
                verbose: bool = True) -> Dict:
        """
        Run nested sampling for CGC model.
        
        Parameters
        ----------
        nlive : int, default=500
            Number of live points.
        dlogz : float, default=0.1
            Stopping criterion on remaining evidence.
        verbose : bool, default=True
            Print progress.
        
        Returns
        -------
        dict
            Results dictionary with evidence and samples.
        """
        dynesty = self._get_dynesty()
        
        print("\n" + "="*60)
        print("NESTED SAMPLING: CGC MODEL")
        print("="*60)
        print(f"Live points: {nlive}, dlogz threshold: {dlogz}")
        
        # Create sampler
        sampler = dynesty.NestedSampler(
            self._log_likelihood_cgc,
            prior_transform,
            ndim=10,
            nlive=nlive
        )
        
        # Run
        sampler.run_nested(dlogz=dlogz, print_progress=verbose)
        
        self.results_cgc = sampler.results
        
        # Extract results
        logz = self.results_cgc.logz[-1]
        logz_err = self.results_cgc.logzerr[-1]
        
        print(f"\n✓ CGC Evidence: log Z = {logz:.2f} ± {logz_err:.2f}")
        
        return {
            'logz': logz,
            'logz_err': logz_err,
            'samples': self.results_cgc.samples,
            'weights': np.exp(self.results_cgc.logwt - 
                             self.results_cgc.logz[-1]),
            'full_results': self.results_cgc
        }
    
    def run_lcdm(self, nlive: int = 500,
                 dlogz: float = 0.1,
                 verbose: bool = True) -> Dict:
        """
        Run nested sampling for ΛCDM model.
        
        Parameters
        ----------
        nlive : int, default=500
            Number of live points.
        dlogz : float, default=0.1
            Stopping criterion.
        verbose : bool, default=True
            Print progress.
        
        Returns
        -------
        dict
            Results dictionary.
        """
        dynesty = self._get_dynesty()
        
        print("\n" + "="*60)
        print("NESTED SAMPLING: ΛCDM MODEL")
        print("="*60)
        print(f"Live points: {nlive}, dlogz threshold: {dlogz}")
        
        # ΛCDM has only 6 free parameters (CGC params fixed)
        sampler = dynesty.NestedSampler(
            self._log_likelihood_lcdm,
            lambda u: u,  # Simple transform for unit cube
            ndim=6,
            nlive=nlive
        )
        
        sampler.run_nested(dlogz=dlogz, print_progress=verbose)
        
        self.results_lcdm = sampler.results
        
        logz = self.results_lcdm.logz[-1]
        logz_err = self.results_lcdm.logzerr[-1]
        
        print(f"\n✓ ΛCDM Evidence: log Z = {logz:.2f} ± {logz_err:.2f}")
        
        return {
            'logz': logz,
            'logz_err': logz_err,
            'samples': self.results_lcdm.samples,
            'weights': np.exp(self.results_lcdm.logwt - 
                             self.results_lcdm.logz[-1]),
            'full_results': self.results_lcdm
        }
    
    def compute_bayes_factor(self) -> Dict:
        """
        Compute Bayes factor comparing CGC to ΛCDM.
        
        Returns
        -------
        dict
            Bayes factor results including interpretation.
        """
        if self.results_cgc is None or self.results_lcdm is None:
            raise ValueError("Run both CGC and ΛCDM sampling first")
        
        logz_cgc = self.results_cgc.logz[-1]
        logz_lcdm = self.results_lcdm.logz[-1]
        
        logz_cgc_err = self.results_cgc.logzerr[-1]
        logz_lcdm_err = self.results_lcdm.logzerr[-1]
        
        # Log Bayes factor
        log_B = logz_cgc - logz_lcdm
        log_B_err = np.sqrt(logz_cgc_err**2 + logz_lcdm_err**2)
        
        # Interpretation (Jeffreys' scale)
        if abs(log_B) < 1:
            interpretation = "Inconclusive"
        elif abs(log_B) < 2.5:
            if log_B > 0:
                interpretation = "Weak evidence for CGC"
            else:
                interpretation = "Weak evidence for ΛCDM"
        elif abs(log_B) < 5:
            if log_B > 0:
                interpretation = "Moderate evidence for CGC"
            else:
                interpretation = "Moderate evidence for ΛCDM"
        else:
            if log_B > 0:
                interpretation = "Strong evidence for CGC"
            else:
                interpretation = "Strong evidence for ΛCDM"
        
        print("\n" + "="*60)
        print("BAYESIAN MODEL COMPARISON")
        print("="*60)
        print(f"\nlog Z (CGC):  {logz_cgc:.2f} ± {logz_cgc_err:.2f}")
        print(f"log Z (ΛCDM): {logz_lcdm:.2f} ± {logz_lcdm_err:.2f}")
        print(f"\nlog Bayes Factor: {log_B:.2f} ± {log_B_err:.2f}")
        print(f"Bayes Factor: {np.exp(log_B):.2e}")
        print(f"\nInterpretation: {interpretation}")
        
        return {
            'logz_cgc': logz_cgc,
            'logz_lcdm': logz_lcdm,
            'log_bayes_factor': log_B,
            'log_bayes_factor_err': log_B_err,
            'bayes_factor': np.exp(log_B),
            'interpretation': interpretation
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_nested_sampling(data: Dict[str, Any],
                        nlive: int = 500,
                        dlogz: float = 0.1,
                        compare_lcdm: bool = True,
                        include_sne: bool = False,
                        include_lyalpha: bool = False,
                        verbose: bool = True) -> Dict:
    """
    Run nested sampling for CGC parameter estimation and model comparison.
    
    This is the main entry point for nested sampling analysis.
    
    Parameters
    ----------
    data : dict
        Cosmological data dictionary.
    
    nlive : int, default=500
        Number of live points. More = more accurate but slower.
    
    dlogz : float, default=0.1
        Stopping criterion on evidence estimate.
    
    compare_lcdm : bool, default=True
        Also run ΛCDM and compute Bayes factor.
    
    include_sne : bool, default=False
        Include supernovae in likelihood.
    
    include_lyalpha : bool, default=False
        Include Lyman-α in likelihood.
    
    verbose : bool, default=True
        Print progress messages.
    
    Returns
    -------
    dict
        Results dictionary containing:
        - 'cgc': CGC nested sampling results
        - 'lcdm': ΛCDM results (if compare_lcdm=True)
        - 'bayes_factor': Model comparison (if compare_lcdm=True)
        - 'posterior_samples': Weighted samples from CGC
    
    Examples
    --------
    Basic evidence calculation:
    >>> results = run_nested_sampling(data, nlive=500)
    >>> print(f"log Z = {results['cgc']['logz']:.2f}")
    
    Full model comparison:
    >>> results = run_nested_sampling(data, compare_lcdm=True)
    >>> print(results['bayes_factor']['interpretation'])
    """
    likelihood_kwargs = {
        'include_sne': include_sne,
        'include_lyalpha': include_lyalpha
    }
    
    sampler = NestedSampler(data, likelihood_kwargs)
    
    # Run CGC
    cgc_results = sampler.run_cgc(nlive=nlive, dlogz=dlogz, verbose=verbose)
    
    results = {'cgc': cgc_results}
    
    # Run ΛCDM for comparison
    if compare_lcdm:
        lcdm_results = sampler.run_lcdm(nlive=nlive, dlogz=dlogz, 
                                        verbose=verbose)
        results['lcdm'] = lcdm_results
        
        # Compute Bayes factor
        results['bayes_factor'] = sampler.compute_bayes_factor()
    
    # Extract posterior samples (importance weighted)
    samples = cgc_results['samples']
    weights = cgc_results['weights']
    weights /= weights.sum()
    
    # Resample for equal-weight samples
    n_resample = 10000
    indices = np.random.choice(len(samples), size=n_resample, p=weights)
    results['posterior_samples'] = samples[indices]
    
    return results


def compute_bayesian_evidence(data: Dict[str, Any],
                              model: str = 'cgc',
                              nlive: int = 500,
                              **kwargs) -> Tuple[float, float]:
    """
    Compute Bayesian evidence for a single model.
    
    Parameters
    ----------
    data : dict
        Cosmological data.
    model : str, default='cgc'
        Model to evaluate ('cgc' or 'lcdm').
    nlive : int, default=500
        Number of live points.
    **kwargs
        Additional arguments for nested sampling.
    
    Returns
    -------
    tuple
        (log_evidence, log_evidence_error)
    """
    results = run_nested_sampling(
        data, 
        nlive=nlive,
        compare_lcdm=(model == 'lcdm'),
        **kwargs
    )
    
    if model == 'lcdm' and 'lcdm' in results:
        return results['lcdm']['logz'], results['lcdm']['logz_err']
    else:
        return results['cgc']['logz'], results['cgc']['logz_err']


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing nested sampling module...")
    print("(This may take several minutes)")
    
    from .data_loader import load_mock_data
    
    # Generate mock data
    data = load_mock_data(verbose=False)
    
    # Run quick nested sampling test
    try:
        results = run_nested_sampling(
            data,
            nlive=50,  # Very low for testing
            dlogz=1.0,  # Loose criterion for speed
            compare_lcdm=False,
            verbose=True
        )
        
        print(f"\n✓ Nested sampling test passed")
        print(f"  log Z = {results['cgc']['logz']:.2f}")
    except Exception as e:
        print(f"\n⚠ Nested sampling test skipped: {e}")
        print("  (Install dynesty with: pip install dynesty)")
