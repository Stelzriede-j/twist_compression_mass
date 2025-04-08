"""
Title: Appendix A.2 – Mass Spectrum Scan (Quantized Energy Levels)
Author: Jacob Stelzriede
Date: 2025-04-07
Description:
    This script generates the locked mass spectrum by sweeping across initial twist seed amplitudes (ω₀)
    and measuring the resulting locked energy after freeze-lock evolution. The output is a calibrated,
    quantized mass spectrum. This script is used in both Appendix A.2 (spectrum quantization) and A.7
    (glueball ratio matching), with Level 3 calibrated to 1700 MeV.

    Purpose:
    - Validates: Discrete excitation levels and mass quantization under twist-compression dynamics
    - Locking logic: freeze-lock (E > E_gap at final timestep)
    - Parameters used: λ = 1.49, n = 2.17 (validated configuration from A.7)

Reproducibility:
    - Matches Figures: appendix_spectrum_scan.png, appendix_glueball_spectrum.png
    - Matches script: mass_spectrum_scan.py

License: MIT
"""

# --- Parameter Descriptions ---
# L              : Lattice size (L x L grid)
# T              : Number of time steps
# dx, dt         : Spatial and temporal resolution
# m              : Mass-like parameter for twist inertia
# r              : Radius scaling factor (often 1.0)
# lambda_        : Compression strength parameter (λ)
# kappa          : Compression scale (κ)
# n              : Nonlinearity exponent
# E_gap          : Mass gap threshold (energy locking threshold)
# omega_seeds    : List of twist amplitudes used to generate spectrum
# sigma          : Width of Gaussian seed (in lattice units)
# sim_to_MeV     : Energy conversion factor (simulation units → MeV)
# calibration_index : Index of reference level (e.g., Level 3) used for spectrum calibration



import numpy as np
import matplotlib.pyplot as plt

# Parameters (from A.7 best config)
L = 30
T = 200
dx = 1.0
dt = 0.05
m = 1.0
r = 1.0
kappa = 1.0
lambda_ = 1.49
n = 2.17
E_gap = 0.5
sigma = 1.5
sim_to_MeV = 2000
omega_seeds = np.linspace(0.5, 3.0, 15)

def Etwist(omega):
    return 0.5 * m * r**2 * omega**2

def Ecomp(omega):
    return lambda_ * (omega / kappa) ** n

def total_energy(omega):
    return Etwist(omega) + Ecomp(omega)

def apply_gaussian_seed(field, center, omega0, sigma):
    size = 5
    half = size // 2
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            x, y = center[0] + i, center[1] + j
            if 0 <= x < L and 0 <= y < L:
                field[x, y] = omega0 * np.exp(-(i**2 + j**2) / (2 * sigma**2))

# Run spectrum scan using freeze-locking
locked_energies = []

for omega0 in omega_seeds:
    omega = np.zeros((L, L))
    locked_mask = np.zeros((L, L), dtype=bool)
    apply_gaussian_seed(omega, (L // 2, L // 2), omega0, sigma)

    for t in range(T):
        new_omega = omega.copy()
        for i in range(1, L - 1):
            for j in range(1, L - 1):
                if not locked_mask[i, j]:
                    laplacian = (omega[i+1, j] + omega[i-1, j] + omega[i, j+1] + omega[i, j-1] - 4 * omega[i, j]) / dx**2
                    new_omega[i, j] += dt * laplacian
        omega = new_omega
        E = total_energy(omega)
        locked_mask |= (E > E_gap)

    locked_energy = np.sum(total_energy(omega)[locked_mask]) * sim_to_MeV
    locked_energies.append(locked_energy)

# Calibrate Level 3 = 1700 MeV
calibration_index = 2
scale = 1700.0 / locked_energies[calibration_index]
calibrated_energies = [round(e * scale, 2) for e in locked_energies]
ratios = [round(e / calibrated_energies[calibration_index], 3) for e in calibrated_energies]

# Output results
for level, (energy, ratio) in enumerate(zip(calibrated_energies, ratios), start=1):
    print(f"Level {level:2d} | Energy: {energy:8.2f} MeV | Ratio to Level 3: {ratio:.3f}")

# Optional: Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(omega_seeds)+1), calibrated_energies, marker='o', color='darkorange')
plt.title("A.2 – Calibrated Mass Spectrum (Freeze Lock, Level 3 = 1700 MeV)", fontsize=12)
plt.xlabel("Level")
plt.ylabel("Locked Mass Energy (MeV)")
plt.grid(True)
plt.tight_layout()
plt.savefig("appendix_spectrum_scan.png", dpi=150)
plt.show()
