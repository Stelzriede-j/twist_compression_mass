"""
Title: Numerical Results 16.5 – Parameter Stability Scan (Volume Cross-Check)
Author: Jacob Stelzriede
Date: 2025-04-07
Description:
    This script sweeps over combinations of twist-compression parameters (λ, n)
    and evaluates their spectral stability across multiple lattice volumes (L = 12, 16, 20, 24).
    It tracks which configurations produce a stable physical mass gap under fixed sampling conditions,
    and identifies statistically consistent regions suitable for further analysis.
    Results from this script support the parameter space discussion in Section 16.5.

    Purpose:
    - Validates: Parameter ranges that consistently yield nonzero mass gaps across volume
    - Locking logic: MCMC-based correlation function decay (not explicit duration or freeze-lock)
    - Parameters scanned: λ and n

Reproducibility:
    - Matches Table: sample-stable configurations (Section 16.5)
    - Matches script: parameter_stability_scan.py

License: MIT
"""

# --- Parameter Descriptions ---
# volumes     : List of lattice volume sizes (e.g., 12, 16, 20, 24)
# a           : Fixed lattice spacing (e.g., a = 0.5)
# num_samples : Number of Metropolis samples per volume
# lambda_vals : Array of λ values to scan
# n_vals      : Array of n values to scan
# seed        : Random seed for reproducibility
# dx, dt      : Grid and time resolution (may be implicit in action)
# fit_window  : Range of correlation decay values used for mass extraction
# m_phys      : Extracted physical mass gap per volume and parameter set
# stable_set  : List of configurations that pass all volume tests


import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import expm
import matplotlib.pyplot as plt
import random

np.random.seed(42)
random.seed(42)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ --- Core SU(2) link generation ---
#~ Generate a small random SU(2) matrix using the exponential map.
#~ Returns a 2x2 complex matrix U ∈ SU(2).
def random_su2(epsilon=0.1):
    #~ Pauli matrices
    sigma = [
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex)
    ] 
    #~ Random unit vector n̂
    n = np.random.randn(3)
    n /= np.linalg.norm(n)
    #~ Random small angle ε * n
    angle = epsilon * np.random.uniform(0.8, 1.2)  # small perturbation
    A = sum(n[i] * sigma[i] for i in range(3))
    #~ Exponential map: U = exp(i * ε * n̂ · σ)
    U = expm(1j * angle * A)
    #~ Ensure unitarity (numerical safety)
    return U / np.linalg.det(U)**0.5


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
#~ Initialize a 4D lattice of SU(2) link variables
#~ Each link U_mu(x) ∈ SU(2) is stored per lattice site and direction
#~ Returns a 7D array: shape [Lx, Ly, Lz, Lt, 4, 2, 2]
def initialize_links(Lx, Ly, Lz, Lt):
    U = np.zeros((Lx, Ly, Lz, Lt, 4, 2, 2), dtype=complex)

    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                for t in range(Lt):
                    for mu in range(4):
                        U[x, y, z, t, mu] = random_su2()

    return U


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- Action and reflection ---
# Compute the trace of the Wilson plaquette at site x in (mu, nu) plane
# Returns Tr[P_{mu,nu}(x)] to be used in the lattice action
def wilson_plaquette(U, x, mu, nu, L):
    Lx, Ly, Lz, Lt = L
    i, j, k, l = x  # site coordinates

    #~ Shifted indices with periodic BCs
    x_mu = [(i + (mu == 1)) % Lx,
            (j + (mu == 2)) % Ly,
            (k + (mu == 3)) % Lz,
            (l + (mu == 0)) % Lt]

    x_nu = [(i + (nu == 1)) % Lx,
            (j + (nu == 2)) % Ly,
            (k + (nu == 3)) % Lz,
            (l + (nu == 0)) % Lt]

    #~ Gather link matrices
    U1 = U[i, j, k, l, mu]
    U2 = U[tuple(x_mu)][nu]
    U3 = np.conj(U[tuple(x_nu)][mu].T)
    U4 = np.conj(U[i, j, k, l, nu].T)

    #~ Construct plaquette
    P = U1 @ U2 @ U3 @ U4
    return np.trace(P).real


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Perform one Metropolis update at a random site and direction
# Modifies U in-place with probability exp(−ΔS)
def metropolis_step(U, beta=2.0, lam=0.6, kappa=1.0, n=2.5, epsilon=0.24):
    Lx, Ly, Lz, Lt = U.shape[:4]
    x = np.random.randint(Lx)
    y = np.random.randint(Ly)
    z = np.random.randint(Lz)
    t = np.random.randint(Lt)
    mu = np.random.randint(4)

    #~ Propose SU(2) update: R near identity
    R = random_su2(epsilon)

    #~ Current link
    U_old = U[x, y, z, t, mu]
    U_new = R @ U_old

    #~ Compute local action change ΔS
    delta = 0.0

    for nu in range(4):
        if nu == mu:
            continue

        #~ Get affected plaquette before update
        trP_old = wilson_plaquette(U, (x, y, z, t), mu, nu, (Lx, Ly, Lz, Lt))
        delta_old = (1 - 0.5 * trP_old)
        comp_old = lam * (delta_old / (kappa ** 2)) ** (n / 2)

        #~ Temporarily apply update
        U[x, y, z, t, mu] = U_new

        trP_new = wilson_plaquette(U, (x, y, z, t), mu, nu, (Lx, Ly, Lz, Lt))
        delta_new = (1 - 0.5 * trP_new)
        comp_new = lam * (delta_new / (kappa ** 2)) ** (n / 2)

        #~ Accumulate ΔS
        delta += beta * (delta_new - delta_old) + (comp_new - comp_old)

        #~ Restore original link (for now)
        U[x, y, z, t, mu] = U_old

    #~ Accept/reject
    if np.random.rand() < np.exp(-delta):
        U[x, y, z, t, mu] = U_new
        return True
    return False


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run a fixed number of Metropolis steps to thermalize the lattice
# Modifies U in-place. Returns number of accepted updates.
def thermalize(U, steps=1000, beta=2.0, lam=0.6, kappa=1.0, n=2.5, epsilon=0.24):
    accepted = 0
    for _ in range(steps):
        accepted += int(metropolis_step(U, beta, lam, kappa, n, epsilon))
    return accepted


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Measure the local observable O(x) at a site x = (i, j, k, l)
# Observable: sum of real traces of all plaquettes at that site
def measure_observable(U, x):
    Lx, Ly, Lz, Lt = U.shape[:4]
    i, j, k, l = x
    total = 0.0

    for mu in range(4):
        for nu in range(mu + 1, 4):
            trP = wilson_plaquette(U, (i, j, k, l), mu, nu, (Lx, Ly, Lz, Lt))
            total += trP

    return total

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Compute spatial 2-point function C(r) = ⟨O(x₀) O(x₀ + r·ê)⟩
# along a given axis (0=x, 1=y, 2=z) for fixed time slice
def compute_correlation(U, x0, axis=0):
    Lx, Ly, Lz, Lt = U.shape[:4]
    L = [Lx, Ly, Lz, Lt]
    max_r = L[axis]

    C = np.zeros(max_r)
    O0 = measure_observable(U, x0)

    for r in range(max_r):
        x = list(x0)
        x[axis] = (x[axis] + r) % L[axis]
        O_r = measure_observable(U, tuple(x))
        C[r] = O0 * O_r

    return C


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
# Apply simple 3-point smoothing filter to reduce noise
def smooth(y):
    return np.convolve(y, np.ones(3)/3, mode='valid')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
def test_parameter_combo(lam, n, a=0.5, volumes=[12, 16, 20, 24], N_samples=10000):
    fit_start = 5
    fit_end = 10
    beta = 2.0
    kappa = 1.0
    epsilon = 0.24

    passed = True
    results = {}

    for L_size in volumes:
        L = (L_size, L_size, L_size, L_size * 2)
        U = initialize_links(*L)
        thermalize(U, steps=3000, beta=beta, lam=lam, kappa=kappa, n=n, epsilon=epsilon)

        x0 = tuple(l // 2 for l in L)
        C_sum = np.zeros(L[0])
        O_sum = 0.0

        for _ in range(N_samples):
            for _ in range(5):
                metropolis_step(U, beta, lam, kappa, n, epsilon)
            C_r = compute_correlation(U, x0, axis=0)
            C_sum += C_r
            O_sum += measure_observable(U, x0)

        C_avg = C_sum / N_samples
        O_avg = O_sum / N_samples
        C_conn = C_avg - O_avg**2
        C_norm = np.abs(C_conn / C_conn[0])

        C_fit = C_norm[fit_start:fit_end + 1]
        if len(C_fit) < 5:
            passed = False
            break

        C_smoothed = smooth(C_fit)
        if len(C_smoothed) < 3:
            passed = False
            break

        r_fit = np.arange(fit_start + 1, fit_start + 1 + len(C_smoothed))
        log_C = np.log(C_smoothed)

        def linear_model(r, a, b): return a * r + b
        try:
            params, _ = curve_fit(linear_model, r_fit, log_C)
            slope, _ = params
            xi = -1 / slope
            m_lat = 1 / xi
            m_phys = m_lat / a
            results[L_size] = m_phys
            if m_phys <= 0:
                passed = False
        except:
            passed = False

    return passed, lam, n, results


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    num_trials = 50
    found = []

    for i in range(num_trials):
        lam = round(random.uniform(1.0, 1.7), 3)
        n = round(random.uniform(3.0, 4.7), 2)

        print(f"\n[{i+1}/{num_trials}] Testing λ = {lam}, n = {n} ...")
        passed, lam, n, results = test_parameter_combo(lam, n)

        if passed:
            print("  ✅ PASS — All m_phys > 0")
            for L, m in results.items():
                print(f"    L = {L} → m_phys = {m:.4f}")
            found.append((lam, n, results))
        else:
            print("  ❌ FAIL — unstable or negative at some L")

    print("\n--- Summary of Passing Parameter Sets ---")
    for lam, n, res in found:
        print(f"λ = {lam}, n = {n}, m_phys = {[f'{res[L]:.4f}' for L in sorted(res)]}")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    main()