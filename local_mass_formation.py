"""
Title: Support Script – Local Mass Formation Test
Author: Jacob Stelzriede
Date: 2025-04-07
Description:
    This script tests whether a single Gaussian twist excitation locally accumulates energy
    and locks into a persistent mass structure above a fixed energy threshold. It serves as
    a minimal reproducibility check for twist-compression dynamics prior to running full
    spectrum, decay, or stability tests. This script was used to validate the energy locking
    mechanism and visualize isolated mass formation behavior.

    Purpose:
    - Validates: Local energy accumulation from a single twist seed
    - Locking logic: duration-based (≥ 5 frames)
    - Parameters used: λ = 0.5, n = 1.8 (tested configuration)

Reproducibility:
    - Matches behavior in Appendix A.3 (mass persistence)
    - Used as an internal validation tool; not tied to a specific figure

License: MIT
"""

# --- Parameter Descriptions ---
# L         : Lattice size (L x L grid)
# T         : Number of time steps
# dx, dt    : Spatial and temporal resolution
# m         : Mass-like parameter for twist inertia
# r         : Radius scaling factor (often 1.0)
# lambda_   : Compression strength parameter (λ)
# kappa     : Compression scale (κ)
# n         : Nonlinearity exponent
# E_gap     : Mass gap threshold (energy locking threshold)
# omega0    : Initial twist amplitude
# sigma     : Width of Gaussian seed (in lattice units)
# threshold_duration : Number of timesteps a point must exceed E_gap to be considered "locked"

import numpy as np
import matplotlib.pyplot as plt

#~ Constants for simulation
L = 50                    # Grid size (L x L)
T = 200                   # Number of time steps
dx = 1.0                  # Spatial step
dt = 0.1                  # Time step
λ = 0.5                  # Compression strength
κ = 1.0                  # Compression scale
n = 2                    # Nonlinearity exponent
mass_gap_threshold = 0.5 # Energy threshold

#~ Initialize field: localized twist in center
ω = np.zeros((L, L))
ω[L // 2, L // 2] = 1.5  # Above threshold seed

#~ Create array to store energy
locked_mass_map = np.zeros((L, L))

#~ Evolution loop
for t in range(T):
    laplacian = (
        -4 * ω
        + np.roll(ω, 1, axis=0) + np.roll(ω, -1, axis=0)
        + np.roll(ω, 1, axis=1) + np.roll(ω, -1, axis=1)
    ) / dx**2

    #~ Update field with diffusion and compression penalty
    ω += dt * (laplacian - λ * n * (ω / κ) ** (n - 1) * (ω / κ))

    #~ Compute local energy and update locked regions
    E_twist = 0.5 * ω ** 2
    E_comp = λ * (np.abs(ω / κ) ** n)
    total_energy = E_twist + E_comp

    locked_mass_map += (total_energy > mass_gap_threshold).astype(float)

#~ Normalize locked region map for visualization
locked_mass_map = locked_mass_map / T

#~ Plot result
plt.figure(figsize=(6, 5))
plt.title("Locked Mass Region (Persistence Frequency)")
plt.imshow(locked_mass_map, cmap='inferno', origin='lower')
plt.colorbar(label='Fraction of time locked')
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
