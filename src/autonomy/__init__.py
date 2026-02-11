"""
autonomy - Autonomous GNC Subsystems for Spacecraft Mission Operations

This package implements machine-learning and reinforcement-learning based
autonomy modules that augment traditional GNC algorithms:

    anomaly_detection   : ML-based sensor health monitoring using autoencoders
                          and statistical methods (Mahalanobis distance).
    auto_trajectory     : Reinforcement-learning (Q-learning) agent for
                          autonomous trajectory correction maneuver planning.
    attitude_predictor  : Feedforward neural network for rapid attitude state
                          prediction, complementing numerical propagators.

All neural network code is implemented in pure NumPy to keep the package
self-contained and to demonstrate first-principles understanding of the
underlying mathematics.
"""
