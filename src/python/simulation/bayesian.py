"""
===============================================================================
Bayesian Parameter Estimation and Inference
===============================================================================
Estimates uncertain spacecraft parameters from observations using Bayesian
methods. Implements Gaussian conjugate updates (fast, closed-form) and
Markov Chain Monte Carlo (MCMC) via Metropolis-Hastings for non-Gaussian
posteriors.

Parameters estimated:
    - Spacecraft mass (from F=ma observations)
    - Drag coefficient (from deceleration observations)
    - Thrust coefficient (from acceleration observations)
    - Specific impulse (from propellant consumption rate)

Bayesian framework:
    Prior:      p(theta) ~ N(mu_prior, sigma_prior^2)
    Likelihood: p(y|theta) ~ N(h(theta), sigma_likelihood^2)
    Posterior:  p(theta|y) proportional to p(y|theta) * p(theta)

For Gaussian conjugate priors, the posterior is also Gaussian:
    mu_post = (sigma_lik^2 * mu_prior + sigma_prior^2 * y) /
              (sigma_prior^2 + sigma_lik^2)
    sigma_post^2 = (sigma_prior^2 * sigma_lik^2) /
                   (sigma_prior^2 + sigma_lik^2)

References:
    - Gelman et al., "Bayesian Data Analysis," 3rd ed., 2013
    - Robert & Casella, "Monte Carlo Statistical Methods," 2004
===============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Callable, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BayesianEstimator:
    """
    Bayesian parameter estimator for spacecraft system identification.

    Maintains prior/posterior beliefs about uncertain parameters
    and updates them as observations become available during the mission.

    Args:
        prior_means: Array of prior mean values for each parameter
        prior_sigmas: Array of prior standard deviations
        param_names: Optional list of parameter names for logging
    """

    def __init__(self, prior_means: np.ndarray, prior_sigmas: np.ndarray,
                 param_names: Optional[list] = None):
        self.n_params = len(prior_means)
        self.means = np.array(prior_means, dtype=float)
        self.sigmas = np.array(prior_sigmas, dtype=float)

        # Store initial priors for comparison
        self.prior_means = self.means.copy()
        self.prior_sigmas = self.sigmas.copy()

        # Parameter names for reporting
        if param_names is None:
            self.param_names = [f"param_{i}" for i in range(self.n_params)]
        else:
            self.param_names = param_names

        # History of updates
        self.update_history = []
        self.n_updates = 0

    def update(self, observation: float, predicted: float,
               likelihood_sigma: float, param_index: int = 0):
        """
        Bayesian update for a single parameter using Gaussian conjugate prior.

        Given an observation y and predicted value h(theta), update the
        posterior distribution of the parameter.

        Posterior mean:
            mu_post = (sigma_lik^2 * mu_prior + sigma_prior^2 * y) /
                      (sigma_prior^2 + sigma_lik^2)

        Posterior variance:
            sigma_post^2 = (sigma_prior^2 * sigma_lik^2) /
                           (sigma_prior^2 + sigma_lik^2)

        Args:
            observation: Observed value (y)
            predicted: Model-predicted value (not used in conjugate, but logged)
            likelihood_sigma: Standard deviation of the likelihood
            param_index: Which parameter to update (default 0)
        """
        i = param_index
        sigma_prior_sq = self.sigmas[i] ** 2
        sigma_lik_sq = likelihood_sigma ** 2

        # Gaussian conjugate update
        denom = sigma_prior_sq + sigma_lik_sq

        new_mean = (sigma_lik_sq * self.means[i] +
                    sigma_prior_sq * observation) / denom
        new_sigma_sq = (sigma_prior_sq * sigma_lik_sq) / denom

        self.means[i] = new_mean
        self.sigmas[i] = np.sqrt(new_sigma_sq)

        self.update_history.append({
            'step': self.n_updates,
            'param': self.param_names[i],
            'observation': observation,
            'predicted': predicted,
            'posterior_mean': new_mean,
            'posterior_sigma': np.sqrt(new_sigma_sq)
        })
        self.n_updates += 1

    def estimate_mass(self, thrust: float, accel_observed: float,
                      accel_sigma: float = 0.01):
        """
        Estimate spacecraft mass from F = m*a.

        mass_observation = thrust / accel_observed

        Args:
            thrust: Known thrust force (N)
            accel_observed: Observed acceleration magnitude (m/s^2)
            accel_sigma: Uncertainty in acceleration measurement
        """
        if abs(accel_observed) < 1e-10:
            return  # Can't estimate if no acceleration

        mass_obs = thrust / accel_observed
        # Propagate uncertainty: sigma_mass = thrust * sigma_a / a^2
        mass_sigma = thrust * accel_sigma / (accel_observed ** 2)

        self.update(mass_obs, self.means[0], mass_sigma, param_index=0)

    def estimate_drag_coefficient(self, deceleration: float, velocity: float,
                                   density: float, area: float, mass: float,
                                   decel_sigma: float = 0.001):
        """
        Estimate drag coefficient from drag deceleration.

        From D = 0.5 * rho * v^2 * Cd * A and a_drag = D/m:
        Cd = 2 * m * a_drag / (rho * v^2 * A)

        Args:
            deceleration: Observed drag deceleration (m/s^2)
            velocity: Current speed (m/s)
            density: Atmospheric density (kg/m^3)
            area: Reference area (m^2)
            mass: Current mass (kg)
            decel_sigma: Uncertainty in deceleration
        """
        denom = density * velocity ** 2 * area
        if abs(denom) < 1e-15:
            return

        cd_obs = 2.0 * mass * deceleration / denom
        cd_sigma = 2.0 * mass * decel_sigma / denom

        # Assuming Cd is param_index=1
        if self.n_params > 1:
            self.update(cd_obs, self.means[1], cd_sigma, param_index=1)

    def run_mcmc(self, log_likelihood_func: Callable,
                 n_samples: int = 5000, burn_in: int = 1000,
                 proposal_sigma: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Markov Chain Monte Carlo via Metropolis-Hastings algorithm.

        For non-Gaussian posteriors where conjugate updates don't apply.

        Algorithm:
        1. Start at current parameter estimates
        2. Propose new parameters: theta* ~ N(theta_current, proposal_sigma)
        3. Compute acceptance ratio: alpha = p(y|theta*)*p(theta*) /
                                             (p(y|theta)*p(theta))
        4. Accept with probability min(1, alpha)
        5. Repeat

        Args:
            log_likelihood_func: Function(params) -> log(p(y|theta))
            n_samples: Total number of MCMC samples
            burn_in: Number of initial samples to discard (warm-up)
            proposal_sigma: Proposal distribution std dev per parameter.
                           Defaults to 10% of current sigma.

        Returns:
            DataFrame with columns for each parameter plus 'log_likelihood',
            'accepted' (boolean), after burn-in removal
        """
        if proposal_sigma is None:
            proposal_sigma = self.sigmas * 0.1

        # Initialize chain at current estimate
        current = self.means.copy()
        current_ll = log_likelihood_func(current)
        current_lp = self._log_prior(current)

        # Storage
        chain = np.zeros((n_samples, self.n_params))
        log_likelihoods = np.zeros(n_samples)
        accepted = np.zeros(n_samples, dtype=bool)
        n_accepted = 0

        for i in range(n_samples):
            # Propose new parameters (random walk)
            proposed = current + proposal_sigma * np.random.randn(self.n_params)

            # Compute log-posterior for proposed parameters
            proposed_ll = log_likelihood_func(proposed)
            proposed_lp = self._log_prior(proposed)

            # Log acceptance ratio (in log space to avoid overflow)
            log_alpha = (proposed_ll + proposed_lp) - (current_ll + current_lp)

            # Accept/reject
            if np.log(np.random.rand()) < log_alpha:
                current = proposed
                current_ll = proposed_ll
                current_lp = proposed_lp
                accepted[i] = True
                n_accepted += 1

            chain[i] = current
            log_likelihoods[i] = current_ll

        # Build DataFrame (exclude burn-in)
        data = {}
        for j in range(self.n_params):
            data[self.param_names[j]] = chain[burn_in:, j]
        data['log_likelihood'] = log_likelihoods[burn_in:]
        data['accepted'] = accepted[burn_in:]

        df = pd.DataFrame(data)

        # Update estimates from MCMC chain
        for j in range(self.n_params):
            self.means[j] = np.mean(chain[burn_in:, j])
            self.sigmas[j] = np.std(chain[burn_in:, j])

        acceptance_rate = n_accepted / n_samples
        logger.info(f"MCMC complete: {n_samples} samples, "
                    f"acceptance rate: {acceptance_rate:.1%}")

        return df

    def _log_prior(self, params: np.ndarray) -> float:
        """
        Compute log of prior probability.

        Assumes independent Gaussian priors for each parameter.
        log p(theta) = sum_i -0.5 * ((theta_i - mu_i) / sigma_i)^2 + const
        """
        normalized = (params - self.prior_means) / self.prior_sigmas
        return -0.5 * np.sum(normalized ** 2)

    def get_posterior_mean(self) -> np.ndarray:
        """Return current posterior mean estimates."""
        return self.means.copy()

    def get_posterior_sigma(self) -> np.ndarray:
        """Return current posterior standard deviations."""
        return self.sigmas.copy()

    def get_confidence_interval(self, param_index: int,
                                 confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute symmetric confidence interval for a parameter.

        For Gaussian: CI = mean +/- z * sigma
        where z = norm.ppf((1 + confidence) / 2)

        Args:
            param_index: Which parameter
            confidence: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            Tuple (lower_bound, upper_bound)
        """
        z = stats.norm.ppf((1.0 + confidence) / 2.0)
        lower = self.means[param_index] - z * self.sigmas[param_index]
        upper = self.means[param_index] + z * self.sigmas[param_index]
        return (lower, upper)

    def log_marginal_likelihood(self) -> float:
        """
        Compute log marginal likelihood (model evidence).

        For Gaussian model: log p(y) = -0.5 * n * log(2*pi)
            - 0.5 * sum(log(sigma_prior^2 + sigma_lik^2))
            - 0.5 * sum((y - mu_prior)^2 / (sigma_prior^2 + sigma_lik^2))

        Useful for Bayesian model comparison.
        """
        # Approximate from update history
        if len(self.update_history) == 0:
            return 0.0

        log_ml = 0.0
        for entry in self.update_history:
            obs = entry['observation']
            pred_mean = entry.get('predicted', entry['posterior_mean'])
            sigma = entry['posterior_sigma']
            log_ml += stats.norm.logpdf(obs, loc=pred_mean, scale=sigma)

        return log_ml

    def get_update_history(self) -> pd.DataFrame:
        """Return update history as pandas DataFrame."""
        if len(self.update_history) == 0:
            return pd.DataFrame()
        return pd.DataFrame(self.update_history)

    def summary(self) -> str:
        """Print summary of prior vs posterior for all parameters."""
        lines = ["Bayesian Parameter Estimation Summary",
                 "=" * 50]
        for i in range(self.n_params):
            lines.append(f"  {self.param_names[i]}:")
            lines.append(f"    Prior:     {self.prior_means[i]:.4f} +/- "
                        f"{self.prior_sigmas[i]:.4f}")
            lines.append(f"    Posterior: {self.means[i]:.4f} +/- "
                        f"{self.sigmas[i]:.4f}")
            ci = self.get_confidence_interval(i, 0.95)
            lines.append(f"    95% CI:    [{ci[0]:.4f}, {ci[1]:.4f}]")
        lines.append(f"  Total updates: {self.n_updates}")
        return "\n".join(lines)
