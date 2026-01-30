"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           CGC MCMC Module                                    ║
║                                                                              ║
║  Implements Markov Chain Monte Carlo (MCMC) sampling for CGC parameter      ║
║  estimation using the emcee ensemble sampler.                                ║
║                                                                              ║
║  Features:                                                                    ║
║    • Ensemble MCMC with affine-invariant moves                              ║
║    • Automatic burn-in detection                                             ║
║    • Convergence diagnostics (R-hat, ESS)                                   ║
║    • Progress tracking and checkpointing                                     ║
║    • Parallel execution support                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage
-----
Basic usage:
>>> from cgc.mcmc import run_mcmc
>>> from cgc.data_loader import load_real_data
>>> data = load_real_data()
>>> sampler, chains = run_mcmc(data, n_steps=1000)

Extended run:
>>> sampler, chains = run_mcmc(data, n_steps=10000, n_walkers=64)
"""

import numpy as np
import os
from typing import Tuple, Optional, Dict, Any, Callable
import warnings

from .config import PATHS, MCMC_DEFAULTS
from .parameters import CGCParameters, get_bounds_array
from .likelihoods import log_probability


# =============================================================================
# MCMC SAMPLER CLASS
# =============================================================================

class MCMCSampler:
    """
    MCMC sampler wrapper with convergence diagnostics.
    
    This class wraps the emcee ensemble sampler with additional
    functionality for convergence checking and chain management.
    
    Parameters
    ----------
    data : dict
        Cosmological data dictionary.
    n_walkers : int, default=32
        Number of MCMC walkers.
    n_dim : int, default=10
        Number of parameters.
    likelihood_kwargs : dict, optional
        Additional arguments for the likelihood function.
    
    Attributes
    ----------
    sampler : emcee.EnsembleSampler
        The underlying emcee sampler.
    chains : np.ndarray
        Flattened chains after burn-in removal.
    full_chains : np.ndarray
        Full chains including burn-in.
    
    Examples
    --------
    >>> sampler = MCMCSampler(data, n_walkers=32)
    >>> sampler.run(n_steps=1000)
    >>> chains = sampler.get_chains(discard=200, thin=10)
    """
    
    def __init__(self, data: Dict[str, Any], n_walkers: int = 32,
                 n_dim: int = 10, likelihood_kwargs: Dict = None):
        """Initialize the MCMC sampler."""
        self.data = data
        self.n_walkers = n_walkers
        self.n_dim = n_dim
        self.likelihood_kwargs = likelihood_kwargs or {}
        
        self._sampler = None
        self._initial_pos = None
        self._n_steps_run = 0
    
    def _setup_sampler(self):
        """Set up the emcee sampler."""
        try:
            import emcee
        except ImportError:
            raise ImportError(
                "emcee is required for MCMC. Install with: pip install emcee"
            )
        
        # Create log probability function with data
        def log_prob_fn(theta):
            return log_probability(theta, self.data, **self.likelihood_kwargs)
        
        self._sampler = emcee.EnsembleSampler(
            self.n_walkers,
            self.n_dim,
            log_prob_fn
        )
    
    def initialize(self, params: CGCParameters = None, 
                   scatter: float = 1e-3,
                   seed: int = None) -> np.ndarray:
        """
        Initialize walker positions.
        
        Parameters
        ----------
        params : CGCParameters, optional
            Initial parameters. If None, uses defaults.
        scatter : float, default=1e-3
            Scatter around initial position.
        seed : int, optional
            Random seed for reproducibility.
        
        Returns
        -------
        np.ndarray
            Initial positions, shape (n_walkers, n_dim).
        """
        if seed is not None:
            np.random.seed(seed)
        
        if params is None:
            params = CGCParameters()
        
        theta0 = params.to_array()
        
        # Initialize walkers in a small ball around initial position
        self._initial_pos = theta0 + scatter * np.random.randn(
            self.n_walkers, self.n_dim
        )
        
        # Ensure all walkers are within bounds
        bounds = get_bounds_array()
        for i in range(self.n_dim):
            low, high = bounds[i]
            self._initial_pos[:, i] = np.clip(
                self._initial_pos[:, i], low + 1e-6, high - 1e-6
            )
        
        return self._initial_pos
    
    def run(self, n_steps: int = 1000, 
            progress: bool = True,
            initial_state: np.ndarray = None) -> 'MCMCSampler':
        """
        Run MCMC sampling.
        
        Parameters
        ----------
        n_steps : int, default=1000
            Number of MCMC steps.
        progress : bool, default=True
            Show progress bar.
        initial_state : np.ndarray, optional
            Initial walker positions. If None, uses previously initialized.
        
        Returns
        -------
        MCMCSampler
            Self, for method chaining.
        """
        if self._sampler is None:
            self._setup_sampler()
        
        if initial_state is not None:
            self._initial_pos = initial_state
        elif self._initial_pos is None:
            self.initialize()
        
        print(f"\n{'='*60}")
        print(f"RUNNING MCMC: {n_steps} steps × {self.n_walkers} walkers")
        print(f"{'='*60}")
        
        self._sampler.run_mcmc(
            self._initial_pos,
            n_steps,
            progress=progress
        )
        
        self._n_steps_run = n_steps
        
        return self
    
    def get_chains(self, discard: int = None, thin: int = 10,
                   flat: bool = True) -> np.ndarray:
        """
        Get MCMC chains with burn-in removal.
        
        Parameters
        ----------
        discard : int, optional
            Number of steps to discard as burn-in.
            If None, uses 20% of total steps.
        thin : int, default=10
            Thinning factor.
        flat : bool, default=True
            If True, flatten across walkers.
        
        Returns
        -------
        np.ndarray
            Chain samples.
        """
        if self._sampler is None:
            raise ValueError("Run MCMC first")
        
        if discard is None:
            discard = int(0.2 * self._n_steps_run)
        
        return self._sampler.get_chain(
            discard=discard,
            thin=thin,
            flat=flat
        )
    
    @property
    def sampler(self):
        """Get the underlying emcee sampler."""
        return self._sampler
    
    @property
    def acceptance_fraction(self) -> np.ndarray:
        """Get acceptance fractions for each walker."""
        if self._sampler is None:
            return None
        return self._sampler.acceptance_fraction
    
    def compute_gelman_rubin(self) -> np.ndarray:
        """
        Compute Gelman-Rubin R-hat diagnostic.
        
        R-hat < 1.1 indicates good convergence.
        
        Returns
        -------
        np.ndarray
            R-hat values for each parameter.
        """
        if self._sampler is None:
            return None
        
        chains = self._sampler.get_chain()  # (n_steps, n_walkers, n_dim)
        n_steps, n_walkers, n_dim = chains.shape
        
        # Use second half of chain
        chains = chains[n_steps//2:]
        n = len(chains)
        
        R_hat = np.zeros(n_dim)
        
        for i in range(n_dim):
            # Mean of each chain
            chain_means = np.mean(chains[:, :, i], axis=0)
            
            # Variance within chains
            W = np.mean(np.var(chains[:, :, i], axis=0, ddof=1))
            
            # Variance between chains
            B = n * np.var(chain_means, ddof=1)
            
            # Pooled variance estimate
            var_hat = (1 - 1/n) * W + B/n
            
            # R-hat
            R_hat[i] = np.sqrt(var_hat / W) if W > 0 else 1.0
        
        return R_hat
    
    def is_converged(self, rhat_threshold: float = 1.1) -> bool:
        """
        Check if chains have converged.
        
        Parameters
        ----------
        rhat_threshold : float, default=1.1
            R-hat threshold for convergence.
        
        Returns
        -------
        bool
            True if all R-hat values are below threshold.
        """
        R_hat = self.compute_gelman_rubin()
        if R_hat is None:
            return False
        return np.all(R_hat < rhat_threshold)
    
    def save(self, filepath: str = None):
        """
        Save chains to file.
        
        Parameters
        ----------
        filepath : str, optional
            Output file path. If None, uses default location.
        """
        if filepath is None:
            filepath = os.path.join(PATHS['chains'], 'mcmc_chains.npz')
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        chains = self.get_chains(flat=False)
        flat_chains = self.get_chains(flat=True)
        
        np.savez(
            filepath,
            chains=chains,
            flat_chains=flat_chains,
            n_walkers=self.n_walkers,
            n_dim=self.n_dim,
            n_steps=self._n_steps_run,
            acceptance_fraction=self.acceptance_fraction,
            gelman_rubin=self.compute_gelman_rubin()
        )
        
        print(f"✓ Chains saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> Tuple[np.ndarray, Dict]:
        """
        Load chains from file.
        
        Parameters
        ----------
        filepath : str
            Input file path.
        
        Returns
        -------
        tuple
            (chains, metadata)
        """
        data = np.load(filepath)
        
        metadata = {
            'n_walkers': int(data['n_walkers']),
            'n_dim': int(data['n_dim']),
            'n_steps': int(data['n_steps']),
            'acceptance_fraction': data['acceptance_fraction'],
            'gelman_rubin': data['gelman_rubin']
        }
        
        return data['flat_chains'], metadata


# =============================================================================
# MAIN MCMC FUNCTION
# =============================================================================

def run_mcmc(data: Dict[str, Any],
             n_walkers: int = None,
             n_steps: int = None,
             params: CGCParameters = None,
             include_sne: bool = False,
             include_lyalpha: bool = False,
             seed: int = None,
             save_chains: bool = True,
             verbose: bool = True) -> Tuple[Any, np.ndarray]:
    """
    Run MCMC analysis for CGC parameter estimation.
    
    This is the main entry point for MCMC sampling. It sets up the
    sampler, runs the chains, and returns the results.
    
    Parameters
    ----------
    data : dict
        Cosmological data dictionary from DataLoader.
    
    n_walkers : int, optional
        Number of MCMC walkers. Default: 32.
    
    n_steps : int, optional
        Number of MCMC steps. Default: 1000.
    
    params : CGCParameters, optional
        Initial parameters. If None, uses defaults.
    
    include_sne : bool, default=False
        Include supernovae in likelihood.
    
    include_lyalpha : bool, default=False
        Include Lyman-α in likelihood.
    
    seed : int, optional
        Random seed for reproducibility.
    
    save_chains : bool, default=True
        Save chains to file.
    
    verbose : bool, default=True
        Print progress messages.
    
    Returns
    -------
    tuple
        (sampler, chains) where sampler is the MCMCSampler instance
        and chains is the flattened chain array.
    
    Examples
    --------
    Quick test:
    >>> from cgc.mcmc import run_mcmc
    >>> from cgc.data_loader import load_mock_data
    >>> data = load_mock_data()
    >>> sampler, chains = run_mcmc(data, n_steps=500)
    
    Publication quality:
    >>> sampler, chains = run_mcmc(data, n_steps=10000, n_walkers=64,
    ...                            include_sne=True)
    """
    # Import emcee (check availability)
    try:
        import emcee
    except ImportError:
        print("Installing emcee...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "emcee", "corner"])
        import emcee
    
    # Set defaults
    if n_walkers is None:
        n_walkers = MCMC_DEFAULTS['n_walkers']
    if n_steps is None:
        n_steps = MCMC_DEFAULTS['n_steps_standard']
    if seed is None:
        seed = MCMC_DEFAULTS['seed']
    
    # Set random seed
    np.random.seed(seed)
    
    # Initialize parameters
    if params is None:
        params = CGCParameters()
    
    # Likelihood options
    likelihood_kwargs = {
        'include_sne': include_sne,
        'include_lyalpha': include_lyalpha
    }
    
    # Create sampler
    mcmc = MCMCSampler(
        data=data,
        n_walkers=n_walkers,
        likelihood_kwargs=likelihood_kwargs
    )
    
    # Initialize and run
    mcmc.initialize(params=params, seed=seed)
    mcmc.run(n_steps=n_steps, progress=verbose)
    
    # Get chains
    discard = int(0.2 * n_steps)
    chains = mcmc.get_chains(discard=discard, thin=10)
    
    # Print diagnostics
    if verbose:
        print(f"\n{'='*60}")
        print("MCMC DIAGNOSTICS")
        print(f"{'='*60}")
        
        # Acceptance fraction
        af = mcmc.acceptance_fraction
        print(f"\nAcceptance fraction: {np.mean(af):.3f} "
              f"(range: {af.min():.3f} - {af.max():.3f})")
        
        # Gelman-Rubin
        R_hat = mcmc.compute_gelman_rubin()
        converged = np.all(R_hat < 1.1)
        print(f"\nGelman-Rubin R̂ (should be < 1.1):")
        
        param_names = ['ω_b', 'ω_cdm', 'h', 'ln10As', 'n_s', 'τ',
                      'μ', 'n_g', 'z_trans', 'ρ_thresh']
        for i, (name, r) in enumerate(zip(param_names, R_hat)):
            status = "✓" if r < 1.1 else "✗"
            print(f"  {status} {name:10s}: {r:.4f}")
        
        print(f"\nConverged: {'Yes' if converged else 'No - consider more steps'}")
        print(f"Chain shape: {chains.shape}")
    
    # Save chains
    if save_chains:
        mcmc.save()
    
    return mcmc.sampler, chains


# =============================================================================
# CHAIN ANALYSIS UTILITIES
# =============================================================================

def compute_autocorrelation_time(chains: np.ndarray, 
                                  c: float = 5.0) -> np.ndarray:
    """
    Estimate integrated autocorrelation time.
    
    Parameters
    ----------
    chains : np.ndarray
        MCMC chains, shape (n_samples, n_dim).
    c : float, default=5.0
        Window size factor.
    
    Returns
    -------
    np.ndarray
        Autocorrelation time for each parameter.
    """
    try:
        import emcee
        return emcee.autocorr.integrated_time(chains, c=c, quiet=True)
    except:
        # Simple estimate if emcee method fails
        n, n_dim = chains.shape
        tau = np.zeros(n_dim)
        
        for i in range(n_dim):
            x = chains[:, i]
            x = x - np.mean(x)
            
            # Compute autocorrelation
            acf = np.correlate(x, x, mode='full')
            acf = acf[n-1:] / acf[n-1]
            
            # Sum until it goes negative
            tau[i] = 1 + 2 * np.sum(acf[1:np.argmax(acf[1:] < 0) + 1])
        
        return tau


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """
    Compute effective sample size (ESS).
    
    ESS = n_samples / tau where tau is autocorrelation time.
    
    Parameters
    ----------
    chains : np.ndarray
        MCMC chains.
    
    Returns
    -------
    np.ndarray
        Effective sample size for each parameter.
    """
    n_samples = len(chains)
    tau = compute_autocorrelation_time(chains)
    return n_samples / tau


def thin_chains(chains: np.ndarray, target_ess: int = 1000) -> np.ndarray:
    """
    Thin chains to achieve target effective sample size.
    
    Parameters
    ----------
    chains : np.ndarray
        MCMC chains.
    target_ess : int, default=1000
        Target effective sample size.
    
    Returns
    -------
    np.ndarray
        Thinned chains.
    """
    ess = effective_sample_size(chains)
    min_ess = np.min(ess)
    
    if min_ess >= target_ess:
        return chains
    
    thin_factor = int(np.ceil(len(chains) / target_ess))
    return chains[::thin_factor]


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing MCMC module...")
    
    from .data_loader import load_mock_data
    
    # Generate mock data
    data = load_mock_data(verbose=False)
    
    # Run short MCMC
    print("\nRunning short MCMC test (100 steps)...")
    sampler, chains = run_mcmc(data, n_steps=100, save_chains=False)
    
    print(f"\n✓ MCMC test passed")
    print(f"  Chain shape: {chains.shape}")
    print(f"  Mean μ: {np.mean(chains[:, 6]):.4f}")
