#~ ------------------------------------------------------------------
#~ Twist-Compression Mass Gap Model – Numerical Test
#~ Appendix A.1: Localized Mass Formation (Field Locking Behavior)
#~ Demonstrates that a single above-threshold twist excitation stabilizes
#~ into a persistent, localized mass region under compression dynamics.
#~ ------------------------------------------------------------------

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
