import numpy as np
import random

# =========================
# PARÂMETROS GLOBAIS
# =========================
R = 8.314

# Difusão
D0 = 1e-7
EL = 40000

# Temperatura
T_min = 300
phi = 0.1

# Geometria
L = 1e-3
Nx = 50
dx = L / Nx

# Tempo
dt = 0.01
Nt = 1500

# Dataset
MAX_TRAPS = 4
ENERGY_RANGE = (40000, 120000)
DENSITY_RANGE = (0.1, 1.0)

# =========================
# FUNÇÕES FÍSICAS
# =========================
def temperature(t):
    return T_min + phi * t

def diffusion_coefficient(T):
    return D0 * np.exp(-EL / (R * T))

def trap_equilibrium(theta_L, T, traps):
    total = 0
    for trap in traps:
        K = np.exp(-trap["E"] / (R * T))
        denominator = 1 + (K - 1) * theta_L

        if abs(denominator) < 1e-10:
            denominator = 1e-10

        theta_T = (theta_L * K) / denominator
        theta_T = np.clip(theta_T, 0, 1)
        total += trap["density"] * theta_T
    return total

# =========================
# SIMULAÇÃO TDS
# =========================
def simulate_tds(traps):

    C_L = np.ones(Nx)

    temperatures = []
    fluxes = []

    for step in range(Nt):

        t = step * dt
        T = temperature(t)
        D = diffusion_coefficient(T)

        C_new = C_L.copy()

        for i in range(1, Nx-1):

            diffusion = D * (C_L[i+1] - 2*C_L[i] + C_L[i-1]) / (dx**2 + 1e-12)

            theta_L = C_L[i]
            trap_effect = trap_equilibrium(theta_L, T, traps)

            C_new[i] = C_L[i] + dt * (diffusion - 0.01 * trap_effect)
            if np.isnan(C_new[i]) or np.isinf(C_new[i]):
                C_new[i] = 0

        # Condições de contorno
        C_new[0] = 0
        C_new[-1] = 0

        C_L = C_new

        flux = -D * (C_L[1] - C_L[0]) / dx

        temperatures.append(T)
        fluxes.append(flux)

    return np.array(temperatures), np.array(fluxes)

# =========================
# GERAÇÃO DE TRAPS
# =========================
def generate_random_traps():
    n_traps = random.randint(1, MAX_TRAPS)

    traps = []
    for _ in range(n_traps):
        E = random.uniform(*ENERGY_RANGE)
        density = random.uniform(*DENSITY_RANGE)

        traps.append({"E": E, "density": density})

    return n_traps, traps

# =========================
# PRÉ-PROCESSAMENTO
# =========================
def resample_curve(T, flux, n_points=64):
    T_new = np.linspace(T.min(), T.max(), n_points)
    flux_new = np.interp(T_new, T, flux)
    return flux_new

def preprocess(flux):
    flux = np.maximum(flux, 1e-10)
    flux = np.log(flux)

    std = np.std(flux)
    if std < 1e-10:
        std = 1e-10

    flux = (flux - np.mean(flux)) / std
    return flux

def format_output(n_traps, traps):
    energies = [t["E"] for t in traps]
    densities = [t["density"] for t in traps]

    while len(energies) < MAX_TRAPS:
        energies.append(0)
        densities.append(0)

    return n_traps, energies, densities

# =========================
# DATASET
# =========================
def generate_dataset(N):

    X = []
    y_traps = []
    y_E = []
    y_D = []

    for i in range(N):

        n_traps, traps = generate_random_traps()

        T, flux = simulate_tds(traps)

        flux = resample_curve(T, flux)
        flux = preprocess(flux)

        n, E, D = format_output(n_traps, traps)

        X.append(flux)
        y_traps.append(n)
        y_E.append(E)
        y_D.append(D)

        if i % 100 == 0:
            print(f"{i}/{N}")
    X = np.array(X)
    y_traps = np.array(y_traps)
    y_E = np.array(y_E) / 120000
    y_D = np.array(y_D)

    return X, y_traps, y_E, y_D

# =========================
# EXECUÇÃO
# =========================
if __name__ == "__main__":

    N_SAMPLES = 2000  # ajuste conforme necessário

    X, y_traps, y_E, y_D = generate_dataset(N_SAMPLES)

    np.save("X.npy", X)
    np.save("y_traps.npy", y_traps)
    np.save("y_E.npy", y_E)
    np.save("y_D.npy", y_D)

    print("Dataset gerado com sucesso!")