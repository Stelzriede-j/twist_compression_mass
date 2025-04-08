"""
Title: Appendix A.5 – Decay Test (Persistence Without Continued Input)
Author: Jacob Stelzriede
Date: 2025-04-07
Description:
    This script implements the test described in Appendix A.5 of the paper:
    "Twist Compression and the Yang–Mills Mass Gap"

    Purpose:
    - Validates: Mass persistence after twist propagation halts
    - Locking logic: duration-based (≥ 5 frames)
    - Parameters used: λ = 0.5, n = 1.8 (validated configuration from freeze-lock spectrum)

Reproducibility:
    - Matches Figure: appendix_decay_test.png
    - Matches script: a5_decay_test.py

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
#



import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 50
T = 300
dx = 1.0
dt = 0.1
m = 1.0
r = 1.0
kappa = 1.0
lambda_ = 0.5
n = 1.8
E_gap = 0.5
omega0 = 1.5
threshold_duration = 5

def total_energy(omega):
    Etwist = 0.5 * m * r**2 * omega**2
    Ecomp = lambda_ * (omega / kappa) ** n
    return Etwist + Ecomp

def apply_gaussian_seed(field, center, omega0, sigma=1.0):
    size = 5
    half = size // 2
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            x, y = center[0] + i, center[1] + j
            if 0 <= x < L and 0 <= y < L:
                field[x, y] = omega0 * np.exp(-(i**2 + j**2) / (2 * sigma**2))

# Initialize field
omega = np.zeros((L, L))
apply_gaussian_seed(omega, center=(L // 2, L // 2), omega0=omega0, sigma=1.0)
lock_counter = np.zeros((L, L), dtype=int)

# Evolve: t < 150 with diffusion; t ≥ 150 frozen
for t in range(T):
    if t < 150:
        new_omega = omega.copy()
        for i in range(1, L - 1):
            for j in range(1, L - 1):
                laplacian = (omega[i+1, j] + omega[i-1, j] + omega[i, j+1] + omega[i, j-1] - 4 * omega[i, j]) / dx**2
                new_omega[i, j] += dt * laplacian
        omega = new_omega

    E = total_energy(omega)
    lock_counter[E > E_gap] += 1

locked_mask = lock_counter >= threshold_duration

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
img = ax.imshow(locked_mask, cmap='gray_r', interpolation='nearest')
ax.set_title("A.5 – Decay Test (Propagation Halts After t = 150)", fontsize=12)
ax.set_xlabel("Grid X", fontsize=10)
ax.set_ylabel("Grid Y", fontsize=10)
ax.tick_params(labelsize=8)

legend_labels = ['Unlocked (White)', 'Locked ≥ 5 Frames (Black)']
colors = ['white', 'black']
handles = [plt.Line2D([0], [0], color=c, linewidth=10) for c in colors]
ax.legend(handles, legend_labels, loc='upper right', fontsize=9, frameon=True)

plt.tight_layout()
plt.savefig("appendix_decay_test.png", dpi=150)
plt.show()
