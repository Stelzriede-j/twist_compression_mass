import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.stats import entropy as spectral_entropy
from scipy.ndimage import label, center_of_mass

# Parameters
Nx, Ny = 50, 50
dt = 0.05
damping = 0.95
lambda_fixed = 1.5
n_fixed = 2.17
mod_p = 17
#injection_times = [0, 50, 100]  # Twist injections at different times
injection_times = [0]
centers = [(10, 10), (40, 10), (25, 40)]  # Spatial centers for twist seeds
#twist_amplitude = 0.75
twist_amplitude = 0.9     # For Ford Circle Alignment
injection_radius = 10.0
threshold = 5  # 3x3 match count to register a local resonance

# Initialize twist field and velocity
phi = np.zeros((Nx, Ny))
velocity = np.zeros((Nx, Ny))

# Evolution loop with timed twist injections
for t in range(150):
    if t in injection_times:
        idx = injection_times.index(t)
        cx, cy = centers[idx]
        for x in range(Nx):
            for y in range(Ny):
                r2 = (x - cx)**2 + (y - cy)**2
                phi[x, y] += np.exp(-r2 / injection_radius) * twist_amplitude

    # Evolve field
    new_phi = phi.copy()
    for x in range(1, Nx - 1):
        for y in range(1, Ny - 1):
            laplacian = (
                phi[x+1, y] + phi[x-1, y] + phi[x, y+1] + phi[x, y-1] - 4 * phi[x, y]
            )
            force = -np.sign(phi[x, y]) * lambda_fixed * n_fixed * (abs(phi[x, y])) ** (n_fixed - 1)
            acceleration = laplacian + force
            velocity[x, y] = damping * (velocity[x, y] + dt * acceleration)
            new_phi[x, y] += velocity[x, y]
    phi = new_phi

# Project to mod p space
mod_field = np.round(phi * 1000).astype(int) % mod_p

# Lock region detection via local pattern alignment
mask = np.zeros_like(mod_field, dtype=bool)
for x in range(1, Nx - 1):
    for y in range(1, Ny - 1):
        patch = mod_field[x-1:x+2, y-1:y+2]
        if np.count_nonzero(patch == mod_field[x, y]) >= threshold:
            mask[x, y] = True

# Label lock zones
labeled_array, num_features = label(mask, structure=np.ones((3, 3), dtype=int))

def compute_zone_energies(phi, labeled_array, lambda_, n):
    zone_energies = []
    zone_centroids = {}
    max_label = np.max(labeled_array)
    for label_id in range(1, max_label + 1):
        mask = (labeled_array == label_id)
        E_twist = 0.5 * phi[mask]**2
        E_comp = lambda_ * np.abs(phi[mask])**n
        E_total = np.sum(E_twist + E_comp)
        zone_energies.append(E_total)
        zone_centroids[label_id] = center_of_mass(mask)
    return zone_centroids, zone_energies

# Parameters used in your force function
lambda_bsd = lambda_fixed
n_bsd = n_fixed

zone_centroids, zone_energies = compute_zone_energies(phi, labeled_array, lambda_bsd, n_bsd)

# Optional: Normalize for comparison
zone_indices = np.arange(1, len(zone_energies) + 1)

print("phi max:", np.max(phi))
print("phi min:", np.min(phi))
print("Max label ID in labeled_array:", np.max(labeled_array))

# --- Normalize BSD Lock Zone Energies ---
zone_energies_norm = (zone_energies - np.min(zone_energies)) / (np.max(zone_energies) - np.min(zone_energies))

# --- Generate Twist-Compression Curve ---
omega_vals = np.linspace(0.5, 3.0, len(zone_energies))
E_twistcomp = 0.5 * omega_vals**2 + lambda_bsd * (omega_vals / 1.0)**n_bsd
E_twistcomp_norm = (E_twistcomp - np.min(E_twistcomp)) / (np.max(E_twistcomp) - np.min(E_twistcomp))

# --- Flattened Ford Circle Height Overlay ---
ford_heights = np.zeros(len(zone_indices))
for q in range(1, 15):
    for p in range(1, q):
        if np.gcd(p, q) == 1:
            pos = int((p / q) * len(zone_indices))
            if 0 <= pos < len(zone_indices):
                r = 1 / (2 * q ** 2)
                ford_heights[pos] += r

ford_norm = (ford_heights - np.min(ford_heights)) / (np.max(ford_heights) - np.min(ford_heights))

# --- Plot All Three Together ---
plt.figure(figsize=(10, 6))
plt.plot(zone_indices, zone_energies_norm, 'o-', label='BSD Lock Zone Energy (normalized)', linewidth=2)
plt.plot(zone_indices, E_twistcomp_norm, '--', label='YM Emergent Nonlinear Term (normalized)', linewidth=2)
plt.plot(zone_indices, ford_norm, ':', label='Flat Ford Circle Height Map (normalized)', linewidth=2)

# --- Labeling & Styling ---
plt.title("Normalized Energy Comparison:\nBSD Lock Zones, Emergent Nonlinear Term, and Ford Geometry", fontsize=14)
plt.xlabel("Zone Index", fontsize=11)
plt.ylabel("Normalized Energy / Height", fontsize=11)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""
# Plot twist field and lock zones
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(mod_field, cmap='viridis')
axs[0].set_title("Twist Field mod 17 (Dynamic Injection)")
axs[0].axis('off')
axs[1].imshow(labeled_array, cmap='tab20')
axs[1].set_title(f"Detected Lock Zones: {num_features}")
axs[1].axis('off')
plt.tight_layout()
plt.show()
"""