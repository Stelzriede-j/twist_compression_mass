#
#  Author: J. Stelzriede
#  
#
# reflection_positivity_test.py


import numpy as np
from scipy.linalg import expm
from numpy.linalg import norm
np.random.seed(42)

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
# Compute the full Euclidean Wilson action over the lattice
# Uses SU(2) trace normalization: Tr[I] = 2
def euclidean_action(U, beta=2.0):
    Lx, Ly, Lz, Lt = U.shape[:4]
    total = 0.0

    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                for l in range(Lt):
                    for mu in range(4):
                        for nu in range(mu + 1, 4):  # mu < nu
                            trP = wilson_plaquette(U, (i, j, k, l), mu, nu, (Lx, Ly, Lz, Lt))
                            total += (1 - 0.5 * trP)

    return beta * total


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Compute Euclidean action with compression term added
# S = β * sum[1 - 0.5 * Tr(P)] + λ * sum[(Tr(1 - P) / κ²) ^ (n/2)]
def euclidean_action_compressed(U, beta=2.0, lam=0.5, kappa=1.0, n=2.5):
    Lx, Ly, Lz, Lt = U.shape[:4]
    S_wilson = 0.0
    S_compression = 0.0

    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                for l in range(Lt):
                    for mu in range(4):
                        for nu in range(mu + 1, 4):
                            trP = wilson_plaquette(U, (i, j, k, l), mu, nu, (Lx, Ly, Lz, Lt))
                            delta = (1 - 0.5 * trP)  # ~ deviation from identity
                            S_wilson += delta
                            S_compression += lam * (delta / (kappa**2))**(n / 2)

    return beta * S_wilson + S_compression

    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Reflect the lattice field in Euclidean time: τ → −τ
# Time-like links are conjugate transposed and reversed
# Space-like links are copied directly to the mirrored site
def reflect_links(U):
    Lx, Ly, Lz, Lt = U.shape[:4]
    U_reflected = np.empty_like(U)

    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                for l in range(Lt):
                    t_reflected = (Lt - 1 - l) % Lt
                    for mu in range(4):
                        if mu == 0:  # time-like link → reverse direction
                            #~ Link now points backward in time
                            U_reflected[i, j, k, t_reflected, mu] = np.conj(U[i, j, k, l, mu].T)
                        else:
                            #~ Spatial link stays the same at reflected site
                            U_reflected[i, j, k, t_reflected, mu] = U[i, j, k, l, mu]

    return U_reflected


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- Test functionals ---
# Define a test functional f[U] as the total plaquette trace at time slice t_slice
# This is a gauge-invariant observable supported only on τ > 0
def test_functional(U, t_slice):
    Lx, Ly, Lz, Lt = U.shape[:4]
    total = 0.0

    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                l = t_slice
                for mu in range(4):
                    for nu in range(mu + 1, 4):
                        trP = wilson_plaquette(U, (i, j, k, l), mu, nu, (Lx, Ly, Lz, Lt))
                        total += trP

    return total.real


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Compute ⟨θf, f⟩ = conjugate[f(θU)] * f(U) * exp(−S[U])
def inner_product(U, f_func, theta_func, S_func):
    f_val = f_func(U)
    U_reflected = theta_func(U)
    f_reflected = f_func(U_reflected)
    S = S_func(U)

    product = np.conj(f_reflected) * f_val * np.exp(-S)
    return product.real

  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- Main driver ---
# Main function to run reflection positivity check on SU(2) lattice
def main():
    #~ Lattice dimensions (Lx, Ly, Lz, Lt)
    L = (8, 8, 8, 16)
    N_samples = 20
    t_slice = L[3] // 2  # Middle time slice

    results = []

    for n in range(N_samples):
        print(f"Sample {n + 1}/{N_samples}")
        U = initialize_links(*L)
        ip = inner_product(
            U,
            lambda u: test_functional(u, t_slice),
            reflect_links,
            lambda u: euclidean_action_compressed(u, beta=2.0, lam=0.6, kappa=1.0, n=2.5)
        )

        results.append(ip)

        if n == 0:  # Just print diagnostics for the first sample
            f_u = test_functional(U, t_slice)
            U_reflected = reflect_links(U)
            f_theta_u = test_functional(U_reflected, t_slice)
            S = euclidean_action_compressed(U, beta=2.0, lam=0.6, kappa=1.0, n=2.5)
            ip = np.conj(f_theta_u) * f_u * np.exp(-S)

            print("\n--- Diagnostics for Sample 1 ---")
            print(f"f(U):           {f_u:.6e}")
            print(f"f(θU):          {f_theta_u:.6e}")
            print(f"Action S[U]:    {S:.6e}")
            print(f"exp(-S):        {np.exp(-S):.6e}")
            print(f"⟨θf, f⟩ value:  {ip.real:.6e}\n")



    results = np.array(results)
    print("\n--- Reflection Positivity Check ---")
    print(f"Number of samples: {N_samples}")
    print(f"Minimum inner product value: {results.min():.6f}")
    print(f"Mean inner product value:    {results.mean():.6f}")
    print(f"Fraction positive:           {(results >= 0).sum()}/{N_samples}")

    

if __name__ == "__main__":
    main()
