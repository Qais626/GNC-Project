"""
auto_trajectory.py - Autonomous Trajectory Correction via Reinforcement Learning
=================================================================================

This module implements a Q-learning agent that decides *when* and *how much*
to fire thrusters for Trajectory Correction Maneuvers (TCMs) during an
interplanetary or cislunar cruise phase.

Background
----------
Traditional TCM planning is done on the ground: the navigation team processes
radiometric tracking data, determines the spacecraft's actual trajectory
relative to the reference, designs a delta-v maneuver, and uplinks the
commands.  This process takes hours to days.

For time-critical missions (e.g. proximity operations, planetary defence,
sample return approach) or communication-denied scenarios, an on-board
autonomous agent that can decide to perform small correction burns is highly
desirable.

Why Reinforcement Learning?
---------------------------
RL naturally handles the *sequential decision problem* that trajectory
correction presents:

* **State**: How far are we from the reference trajectory (position and
  velocity errors) and how much fuel do we have left?
* **Action**: Do nothing, or fire thrusters at one of several magnitude
  levels.
* **Reward**: We want to reach the target (large positive reward), minimise
  fuel consumption (small negative reward per burn), and absolutely avoid
  missing the target (large negative reward).

Q-learning is a model-free, off-policy algorithm that learns an action-value
function Q(s, a) from which the optimal policy can be derived by taking
argmax_a Q(s, a).

Limitations
-----------
* The state space is *discretised*, which limits resolution.  A Deep-Q
  Network (DQN) could work with continuous states but would require a
  neural network (see ``anomaly_detection.SimpleNeuralNet`` for a pure-
  NumPy implementation that could be used here in future work).
* The environment (``SimpleTrajectoryEnv``) is a 1-D simplification.  For
  flight use, a higher-fidelity 3-D dynamics model and a continuous action
  space would be needed.

References
----------
* Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018.
* Watkins & Dayan, "Q-learning", *Machine Learning*, 1992.
* Gaudet & Furfaro, "Adaptive Guidance and Integrated Navigation with
  Reinforcement Meta-Learning", *Acta Astronautica*, 2020.

Author : GNC Autonomy Team
Date   : 2025
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# ---------------------------------------------------------------------------
#  Simple 1-D Trajectory Environment
# ---------------------------------------------------------------------------

class SimpleTrajectoryEnv:
    """A simplified 1-D trajectory environment for training the RL agent.

    Physics Model
    -------------
    The spacecraft moves along a single axis toward a target at position
    ``target_pos``.  At each time step:

        acceleration = thrust / mass  -  perturbation
        velocity    += acceleration * dt
        position    += velocity * dt
        fuel        -= |thrust| * dt / Isp

    The perturbation represents unmodelled forces (solar radiation pressure,
    third-body gravity, outgassing, etc.) and is drawn from a Gaussian
    distribution each step.

    Episode Termination
    -------------------
    * **Success**: The spacecraft reaches within ``tol`` of the target.
    * **Failure (miss)**: The position exceeds ``max_distance`` from the
      target (overshoot / divergence) or ``max_steps`` is reached.
    * **Failure (fuel)**: Fuel is exhausted.

    Parameters
    ----------
    target_pos : float
        Target position on the 1-D axis (default 1000 km, stored in km).
    init_pos : float
        Initial position (default 0 km).
    init_vel : float
        Initial velocity toward target (default 1.0 km/s).
    fuel : float
        Initial fuel mass (default 100 kg).
    mass : float
        Dry mass of the spacecraft (default 500 kg).
    dt : float
        Time step in seconds (default 10 s).
    max_steps : int
        Maximum steps per episode.
    perturbation_std : float
        Std of the random perturbation acceleration (km/s^2).
    tol : float
        Distance tolerance for "arrival" at target (km).
    """

    # Actions map to thrust magnitudes (km/s^2 equivalent, simplified)
    ACTION_THRUST = {
        0: 0.0,       # no burn
        1: 0.001,     # small correction
        2: 0.005,     # medium correction
        3: 0.02,      # large correction
    }

    def __init__(
        self,
        target_pos: float = 1000.0,
        init_pos: float = 0.0,
        init_vel: float = 1.0,
        fuel: float = 100.0,
        mass: float = 500.0,
        dt: float = 10.0,
        max_steps: int = 2000,
        perturbation_std: float = 0.0002,
        tol: float = 5.0,
        seed: Optional[int] = None,
    ) -> None:
        self.target_pos = target_pos
        self.init_pos = init_pos
        self.init_vel = init_vel
        self.init_fuel = fuel
        self.mass = mass
        self.dt = dt
        self.max_steps = max_steps
        self.perturbation_std = perturbation_std
        self.tol = tol
        self.rng = np.random.default_rng(seed)

        # Current state
        self.pos = init_pos
        self.vel = init_vel
        self.fuel = fuel
        self.step_count = 0

    def reset(self) -> Tuple[float, float, float]:
        """Reset the environment to initial conditions.

        Returns
        -------
        state : tuple (position_error, velocity_error_proxy, fuel_remaining)
            The initial observation.  ``velocity_error_proxy`` is the
            deviation of the current velocity from an idealised constant-
            velocity transfer.
        """
        self.pos = self.init_pos
        self.vel = self.init_vel
        self.fuel = self.init_fuel
        self.step_count = 0
        return self._get_state()

    def _get_state(self) -> Tuple[float, float, float]:
        """Pack the environment state into the observation tuple."""
        position_error = abs(self.target_pos - self.pos)
        # Ideal velocity to reach target in remaining "time budget"
        remaining_steps = max(self.max_steps - self.step_count, 1)
        ideal_vel = position_error / (remaining_steps * self.dt)
        velocity_error = abs(self.vel - ideal_vel)
        return (position_error, velocity_error, self.fuel)

    def step(self, action: int) -> Tuple[Tuple[float, float, float], float, bool]:
        """Advance the environment by one time step.

        Parameters
        ----------
        action : int in {0, 1, 2, 3}
            Index into ``ACTION_THRUST``.

        Returns
        -------
        next_state : tuple
        reward : float
        done : bool
        """
        assert action in self.ACTION_THRUST, f"Invalid action: {action}"

        thrust = self.ACTION_THRUST[action]
        fuel_cost = thrust * self.dt * 50.0  # simplified fuel model

        # Check if we have enough fuel
        if fuel_cost > self.fuel:
            thrust = 0.0
            fuel_cost = 0.0

        # Apply thrust (always toward the target for simplicity)
        direction = 1.0 if self.target_pos > self.pos else -1.0
        accel = direction * thrust / (self.mass + self.fuel)

        # Random perturbation (unmodelled forces)
        perturbation = self.rng.normal(0, self.perturbation_std)

        # Integrate kinematics (simple Euler)
        self.vel += (accel + perturbation) * self.dt
        self.pos += self.vel * self.dt
        self.fuel -= fuel_cost
        self.fuel = max(self.fuel, 0.0)
        self.step_count += 1

        # --- Reward calculation ---
        position_error = abs(self.target_pos - self.pos)
        done = False
        reward = 0.0

        # Fuel penalty: every burn costs a little reward
        if action > 0 and fuel_cost > 0:
            reward -= 1.0 * action  # larger burns cost more

        # Success: reached the target
        if position_error < self.tol:
            reward += 100.0
            done = True

        # Failure: overshot or diverged
        elif self.pos > self.target_pos + 200.0 or self.pos < -200.0:
            reward -= 1000.0
            done = True

        # Failure: ran out of time
        elif self.step_count >= self.max_steps:
            # Partial reward based on how close we got
            reward -= min(position_error, 1000.0)
            done = True

        # Failure: ran out of fuel far from target
        elif self.fuel <= 0.0 and position_error > 50.0:
            reward -= 500.0
            done = True

        next_state = self._get_state()
        return next_state, reward, done


# ---------------------------------------------------------------------------
#  Q-Learning Trajectory Corrector
# ---------------------------------------------------------------------------

@dataclass
class TrainingMetrics:
    """Stores per-episode training statistics."""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    episode_successes: List[bool] = field(default_factory=list)
    episode_fuel_remaining: List[float] = field(default_factory=list)

    def to_dataframe(self):
        """Convert to a pandas DataFrame if pandas is available."""
        if not _HAS_PANDAS:
            raise ImportError("pandas is required for to_dataframe()")
        return pd.DataFrame({
            "episode": list(range(len(self.episode_rewards))),
            "total_reward": self.episode_rewards,
            "length": self.episode_lengths,
            "success": self.episode_successes,
            "fuel_remaining": self.episode_fuel_remaining,
        })


class TrajectoryCorrector:
    """Q-learning agent for autonomous trajectory correction maneuvers.

    Q-Learning Algorithm
    --------------------
    Q-learning maintains a table Q(s, a) that estimates the expected
    cumulative discounted reward for taking action *a* in state *s* and
    then following the optimal policy.  The update rule is:

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

    where:
        * alpha  = learning rate (how fast we update)
        * gamma  = discount factor (how much we value future rewards)
        * r      = immediate reward
        * s'     = next state after taking action a in state s

    Epsilon-Greedy Exploration
    --------------------------
    With probability epsilon we choose a random action (explore), otherwise
    we choose the action with the highest Q-value (exploit).  Epsilon is
    typically annealed during training so the agent explores more early on
    and exploits more as it learns.

    State Discretisation
    --------------------
    Since we use a tabular Q-table (not a neural network), we must convert
    the continuous state (position_error, velocity_error, fuel_remaining)
    into discrete bins.  The ``state_bins`` parameter defines the bin edges
    for each dimension.

    Parameters
    ----------
    state_bins : list of ndarray
        Bin edges for each state dimension.  For example:
        [np.linspace(0, 1000, 20),   # position error bins
         np.linspace(0, 1, 15),      # velocity error bins
         np.linspace(0, 100, 10)]    # fuel remaining bins
    n_actions : int
        Number of discrete actions (default 4: no-burn, small, medium, large).
    alpha : float
        Learning rate for Q-table updates (default 0.1).
    gamma : float
        Discount factor (default 0.99).  Values close to 1 make the agent
        far-sighted; values close to 0 make it myopic.
    epsilon : float
        Initial exploration rate (default 0.1).
    epsilon_decay : float
        Multiplicative decay applied to epsilon each episode (default 0.999).
    epsilon_min : float
        Minimum exploration rate (default 0.01).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        state_bins: Optional[List[np.ndarray]] = None,
        n_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        # Default bins if none provided
        if state_bins is None:
            state_bins = [
                np.linspace(0, 1200, 25),    # position error (km)
                np.linspace(0, 2.0, 15),     # velocity error (km/s)
                np.linspace(0, 100, 12),      # fuel remaining (kg)
            ]

        self.state_bins = state_bins
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed)

        # Compute the total number of discrete states.
        # Each dimension has (len(bins) - 1) bins, but np.digitize can also
        # return 0 or len(bins), so we add 1 for overflow on each side.
        self._n_bins = [len(b) + 1 for b in state_bins]
        self._total_states = int(np.prod(self._n_bins))

        # Initialise the Q-table to small random values.
        # Random init breaks symmetry so the agent doesn't start by always
        # choosing the same action.
        self.q_table = self.rng.uniform(
            low=-0.01,
            high=0.01,
            size=(self._total_states, n_actions),
        )

        # Training metrics
        self.metrics = TrainingMetrics()

    # ---- State discretisation ------------------------------------------------

    def discretize_state(
        self,
        position_error: float,
        velocity_error: float,
        fuel_remaining: float,
    ) -> int:
        """Convert continuous state to a single integer index into the Q-table.

        Each continuous value is digitised into the corresponding bin array,
        then the multi-dimensional index is flattened into a single integer
        using C-order (row-major) ravelling.

        Parameters
        ----------
        position_error : float
        velocity_error : float
        fuel_remaining : float

        Returns
        -------
        int
            Flat index into the Q-table (0 <= index < total_states).
        """
        raw = [position_error, velocity_error, fuel_remaining]
        indices = []
        for val, bins in zip(raw, self.state_bins):
            idx = int(np.digitize(val, bins))
            indices.append(idx)

        # Ravel multi-index to flat index
        flat = 0
        for i, (idx, n) in enumerate(zip(indices, self._n_bins)):
            flat = flat * n + min(idx, n - 1)
        return flat

    def _state_tuple_to_index(self, state: Tuple[float, float, float]) -> int:
        """Convenience wrapper around ``discretize_state``."""
        return self.discretize_state(state[0], state[1], state[2])

    # ---- Policy (action selection) -------------------------------------------

    def choose_action(self, state: Tuple[float, float, float]) -> int:
        """Select an action using the epsilon-greedy policy.

        Epsilon-Greedy
        --------------
        With probability *epsilon* choose a uniformly random action
        (exploration).  Otherwise choose the action with the highest Q-value
        for the current state (exploitation).

        This balance between exploration and exploitation is fundamental to
        RL: too much exploration wastes time on bad actions; too little
        exploration may miss the optimal policy.

        Parameters
        ----------
        state : tuple (position_error, velocity_error, fuel_remaining)

        Returns
        -------
        int
            Action index in {0, ..., n_actions - 1}.
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        else:
            s_idx = self._state_tuple_to_index(state)
            return int(np.argmax(self.q_table[s_idx]))

    def choose_action_greedy(self, state: Tuple[float, float, float]) -> int:
        """Always choose the best-known action (no exploration)."""
        s_idx = self._state_tuple_to_index(state)
        return int(np.argmax(self.q_table[s_idx]))

    # ---- Q-learning update ---------------------------------------------------

    def update_q(
        self,
        state: Tuple[float, float, float],
        action: int,
        reward: float,
        next_state: Tuple[float, float, float],
        done: bool = False,
    ) -> None:
        """Perform a single Q-learning update.

        Update Rule
        -----------
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

        When the episode is done (``done=True``) there is no future reward,
        so the target simplifies to just ``r``.

        Parameters
        ----------
        state : tuple
        action : int
        reward : float
        next_state : tuple
        done : bool
        """
        s = self._state_tuple_to_index(state)
        s_next = self._state_tuple_to_index(next_state)

        # Temporal difference target
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[s_next])

        # TD error: how surprised are we?
        td_error = td_target - self.q_table[s, action]

        # Update Q-value toward the target
        self.q_table[s, action] += self.alpha * td_error

    # ---- Training loop -------------------------------------------------------

    def train(
        self,
        env: SimpleTrajectoryEnv,
        num_episodes: int = 1000,
        verbose: bool = True,
        print_every: int = 100,
    ) -> TrainingMetrics:
        """Train the agent on the trajectory environment.

        For each episode:
        1. Reset the environment.
        2. Repeatedly choose an action, step the environment, and update Q.
        3. Decay epsilon after each episode.

        Parameters
        ----------
        env : SimpleTrajectoryEnv
            The training environment.
        num_episodes : int
            Number of training episodes.
        verbose : bool
            Print progress if True.
        print_every : int
            Print interval (episodes).

        Returns
        -------
        TrainingMetrics
            Per-episode statistics.
        """
        self.metrics = TrainingMetrics()
        t_start = time.time()

        for ep in range(num_episodes):
            state = env.reset()
            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)

                self.update_q(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps += 1

            # Determine success
            pos_error = abs(env.target_pos - env.pos)
            success = pos_error < env.tol

            # Log metrics
            self.metrics.episode_rewards.append(total_reward)
            self.metrics.episode_lengths.append(steps)
            self.metrics.episode_successes.append(success)
            self.metrics.episode_fuel_remaining.append(env.fuel)

            # Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if verbose and (ep % print_every == 0 or ep == num_episodes - 1):
                recent = self.metrics.episode_rewards[-print_every:]
                avg_reward = np.mean(recent)
                recent_succ = self.metrics.episode_successes[-print_every:]
                succ_rate = np.mean(recent_succ) * 100
                print(
                    f"  [TCM Agent] Episode {ep:5d}/{num_episodes}  "
                    f"avg_reward={avg_reward:8.1f}  "
                    f"success_rate={succ_rate:5.1f}%  "
                    f"epsilon={self.epsilon:.4f}"
                )

        elapsed = time.time() - t_start
        if verbose:
            print(f"  [TCM Agent] Training complete in {elapsed:.1f}s")

        return self.metrics

    # ---- Inference (get a correction decision) --------------------------------

    def get_correction(
        self,
        position_error: float,
        velocity_error: float,
        fuel_remaining: float,
    ) -> Tuple[float, float]:
        """Get a trajectory correction recommendation from the trained policy.

        This is the primary interface for operational use.  It takes the
        current navigation state and returns a recommended burn.

        Parameters
        ----------
        position_error : float  (km)
        velocity_error : float  (km/s)
        fuel_remaining : float  (kg)

        Returns
        -------
        burn_magnitude : float
            Recommended thrust magnitude (km/s^2 equivalent).
        direction : float
            +1.0 (toward target) or 0.0 (no burn).

        Notes
        -----
        The agent uses its learned Q-table (greedy policy, no exploration)
        to select the action.  The action is then mapped back to a physical
        thrust magnitude via ``SimpleTrajectoryEnv.ACTION_THRUST``.
        """
        state = (position_error, velocity_error, fuel_remaining)
        action = self.choose_action_greedy(state)

        burn_magnitude = SimpleTrajectoryEnv.ACTION_THRUST[action]
        direction = 1.0 if action > 0 else 0.0

        return burn_magnitude, direction

    # ---- Policy serialisation -------------------------------------------------

    def save_policy(self, filepath: str) -> None:
        """Save the Q-table and configuration to a JSON file.

        Parameters
        ----------
        filepath : str
            Path to the output file.
        """
        path = Path(filepath)
        state = {
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "state_bins": [b.tolist() for b in self.state_bins],
            "q_table": self.q_table.tolist(),
        }
        path.write_text(json.dumps(state, indent=2))

    def load_policy(self, filepath: str) -> None:
        """Load a previously saved Q-table and configuration.

        Parameters
        ----------
        filepath : str
            Path to the saved file.
        """
        path = Path(filepath)
        state = json.loads(path.read_text())

        self.n_actions = state["n_actions"]
        self.alpha = state["alpha"]
        self.gamma = state["gamma"]
        self.epsilon = state["epsilon"]
        self.epsilon_decay = state["epsilon_decay"]
        self.epsilon_min = state["epsilon_min"]
        self.state_bins = [np.array(b) for b in state["state_bins"]]
        self.q_table = np.array(state["q_table"])

        self._n_bins = [len(b) + 1 for b in self.state_bins]
        self._total_states = int(np.prod(self._n_bins))

    # ---- Analysis utilities ---------------------------------------------------

    def q_table_stats(self) -> Dict[str, float]:
        """Return summary statistics of the Q-table for diagnostics."""
        return {
            "mean": float(np.mean(self.q_table)),
            "std": float(np.std(self.q_table)),
            "min": float(np.min(self.q_table)),
            "max": float(np.max(self.q_table)),
            "nonzero_frac": float(np.mean(np.abs(self.q_table) > 0.02)),
            "total_entries": self.q_table.size,
        }

    def evaluate(
        self,
        env: SimpleTrajectoryEnv,
        num_episodes: int = 100,
    ) -> Dict[str, float]:
        """Evaluate the trained policy (greedy, no exploration).

        Parameters
        ----------
        env : SimpleTrajectoryEnv
        num_episodes : int

        Returns
        -------
        dict with keys: success_rate, avg_reward, avg_fuel_remaining, avg_steps
        """
        rewards = []
        successes = []
        fuels = []
        lengths = []

        old_epsilon = self.epsilon
        self.epsilon = 0.0  # disable exploration for evaluation

        for _ in range(num_episodes):
            state = env.reset()
            total_reward = 0.0
            done = False
            steps = 0

            while not done:
                action = self.choose_action(state)
                state, reward, done = env.step(action)
                total_reward += reward
                steps += 1

            pos_error = abs(env.target_pos - env.pos)
            successes.append(pos_error < env.tol)
            rewards.append(total_reward)
            fuels.append(env.fuel)
            lengths.append(steps)

        self.epsilon = old_epsilon  # restore

        return {
            "success_rate": float(np.mean(successes)),
            "avg_reward": float(np.mean(rewards)),
            "avg_fuel_remaining": float(np.mean(fuels)),
            "avg_steps": float(np.mean(lengths)),
        }


# ---------------------------------------------------------------------------
#  Demonstration / self-test
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Quick demonstration of the trajectory correction RL agent."""
    print("=" * 70)
    print("  Autonomous Trajectory Correction - Q-Learning Demo")
    print("=" * 70)

    # Create environment
    env = SimpleTrajectoryEnv(
        target_pos=1000.0,
        init_pos=0.0,
        init_vel=1.0,
        fuel=100.0,
        dt=10.0,
        max_steps=2000,
        perturbation_std=0.0002,
        seed=42,
    )

    # Create agent with default bins
    agent = TrajectoryCorrector(
        n_actions=4,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,         # start with full exploration
        epsilon_decay=0.998,
        epsilon_min=0.01,
        seed=42,
    )

    print(f"\n  State space size: {agent._total_states}")
    print(f"  Action space size: {agent.n_actions}")
    print(f"  Q-table shape: {agent.q_table.shape}")

    # Train
    print("\n--- Training ---")
    metrics = agent.train(env, num_episodes=500, verbose=True, print_every=100)

    # Evaluate
    print("\n--- Evaluation (100 episodes, greedy policy) ---")
    eval_env = SimpleTrajectoryEnv(
        target_pos=1000.0, init_vel=1.0, fuel=100.0, seed=123
    )
    results = agent.evaluate(eval_env, num_episodes=100)
    for key, val in results.items():
        print(f"  {key:25s}: {val:.4f}")

    # Show Q-table stats
    print("\n--- Q-Table Statistics ---")
    for key, val in agent.q_table_stats().items():
        print(f"  {key:25s}: {val:.4f}")

    # Demonstrate get_correction
    print("\n--- Sample Corrections ---")
    test_states = [
        (500.0, 0.5, 80.0),
        (100.0, 0.1, 50.0),
        (10.0, 0.01, 30.0),
        (800.0, 1.0, 10.0),
    ]
    for pos_err, vel_err, fuel in test_states:
        burn, direction = agent.get_correction(pos_err, vel_err, fuel)
        print(
            f"  pos_err={pos_err:7.1f} km  vel_err={vel_err:.2f} km/s  "
            f"fuel={fuel:5.1f} kg  ->  burn={burn:.4f}  dir={direction:.0f}"
        )

    # Show training metrics summary
    if _HAS_PANDAS:
        df = metrics.to_dataframe()
        print("\n--- Training Metrics Summary (pandas) ---")
        print(df.describe().to_string())
    else:
        print("\n  [pandas not available; skipping DataFrame summary]")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
