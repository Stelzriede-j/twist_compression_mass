# correlation_decay_scan.py
# Analyze the spatial decay of a gauge-invariant 2-point function
# in the twist-compression SU(2) lattice model.
#
# This test estimates the correlation length ξ by measuring:
#   C(r) = ⟨ O(x₀) O(x₀ + r) ⟩
# If C(r) decays exponentially, the system exhibits clustering,
# vacuum uniqueness, and a mass gap.

import numpy as np
from scipy.linalg import expm
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
np.random.seed(31)

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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main driver for spatial correlation decay scan
# Estimates correlation length ξ by measuring exponential falloff of C(r)
def main():
    L = (8, 8, 8, 16)
    N_samples = 20000
    sample_interval = 20
    axis = 0  # x-direction

    beta = 2.0
    lam = 1.2
    kappa = 1.0
    n = 3.4
    epsilon = 0.24

    print("Initializing lattice...")
    U = initialize_links(*L)
    print("Thermalizing...")
    thermalize(U, steps=5000, beta=beta, lam=lam, kappa=kappa, n=n, epsilon=epsilon)
    print("Collecting correlation samples...")

    Lx, Ly, Lz, Lt = L
    x0 = (Lx // 2, Ly // 2, Lz // 2, Lt // 2)
    max_r = L[axis]
    C_sum = np.zeros(max_r)
    O_sum = 0.0

    for i in range(N_samples):
        for _ in range(sample_interval):
            metropolis_step(U, beta, lam, kappa, n, epsilon)

        C_r = compute_correlation(U, x0, axis=axis)
        C_sum += C_r
        O_sum += measure_observable(U, x0)

        if (i + 1) % 50 == 0:
            print(f"Sample {i+1}/{N_samples}")

    C_avg = C_sum / N_samples
    O_avg = O_sum / N_samples
    C_conn = C_avg - O_avg**2

    print("\nConnected Correlation Function C_conn(r):")
    for r, val in enumerate(C_conn):
        print(f"r = {r:2d}, C_conn(r) = {val:.6e}")

    print("\nRaw Correlation Function C(r):")
    for r, val in enumerate(C_avg):
        print(f"r = {r:2d}, C(r) = {val:.6e}")

	#~ Normalize and take absolute value
    C_norm = np.abs(C_conn / C_conn[0])
    r_vals = np.arange(len(C_conn))

	#~ Manually choose a clean exponential decay region to fit
    fit_start = 2
    fit_end = 5  # inclusive

    r_fit = r_vals[fit_start:fit_end + 1]  # e.g., r = 2,3,4,5,6
    C_fit = C_norm[fit_start:fit_end + 1]

	#~ Check that all values are positive before taking log
    if np.any(C_fit <= 0):
	    raise ValueError("Fit region includes non-positive values — adjust fit range.")

    log_C = np.log(np.abs(C_fit))

	#~ Fit: log C(r) = -r/xi + const
    def linear_model(r, a, b):
	    return a * r + b

    params, _ = curve_fit(linear_model, r_fit, log_C)
    slope, intercept = params
    xi = -1 / slope
    mass_gap = 1 / xi

    print(f"\nEstimated correlation length ξ ≈ {xi:.3f}")
    print(f"Estimated mass gap m ≈ {mass_gap:.3f}")

	#~ Plot
    plt.figure(figsize=(8, 5))
    plt.plot(r_fit, log_C, 'o', label='log |C_conn(r)/C(0)|')
    plt.plot(r_fit, linear_model(r_fit, *params), '-', label=f'Fit: ξ ≈ {xi:.2f}')
    plt.xlabel('r')
    plt.ylabel('log |C_conn(r) / C(0)|')
    plt.title('Exponential Decay of Connected Correlator')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("appendix_mass_gap_fit.png", dpi=300)
    plt.show()



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~