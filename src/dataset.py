import numpy as np
import os


# ================================
# Configurações globais
# ================================
DEFAULT_N_POINTS = 100
DEFAULT_TEMP_RANGE = (0, 600)


# ================================
# Função de pico gaussiano
# ================================
def gaussian(T, mu, A, sigma):
    return A * np.exp(-(T - mu)**2 / (2 * sigma**2))


# ================================
# Geração de UMA amostra
# ================================
def generate_sample(
    n_points=DEFAULT_N_POINTS,
    temp_range=DEFAULT_TEMP_RANGE,
    n_peaks=1,
    noise_std=0.0
):
    """
    Gera uma curva TDS sintética.

    Retorna:
    - flux: vetor da curva
    - params: lista [mu1, A1, mu2, A2, ...]
    """

    T = np.linspace(temp_range[0], temp_range[1], n_points)
    flux = np.zeros_like(T)

    params = []

    for _ in range(n_peaks):
        mu = np.random.uniform(100, 500)     # posição do pico
        A = np.random.uniform(0.1, 1.0)      # intensidade
        sigma = np.random.uniform(20, 50)    # largura

        flux += gaussian(T, mu, A, sigma)

        params.extend([mu, A])

    # Adicionar ruído (opcional)
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=flux.shape)
        flux += noise

    return flux, np.array(params)


# ================================
# Geração do dataset completo
# ================================
def generate_dataset(
    n_samples=10000,
    n_points=DEFAULT_N_POINTS,
    n_peaks=1,
    noise_std=0.0,
    seed=None
):
    """
    Gera dataset completo.

    Retorna:
    - X: curvas (n_samples, n_points)
    - y: parâmetros (n_samples, 2*n_peaks)
    """

    if seed is not None:
        np.random.seed(seed)

    X = []
    y = []

    for _ in range(n_samples):
        flux, params = generate_sample(
            n_points=n_points,
            n_peaks=n_peaks,
            noise_std=noise_std
        )

        X.append(flux)
        y.append(params)

    return np.array(X), np.array(y)


# ================================
# Normalização (opcional)
# ================================
def normalize_dataset(X):
    """
    Normaliza cada curva entre 0 e 1
    """
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)

    return (X - X_min) / (X_max - X_min + 1e-8)


# ================================
# Salvar dataset
# ================================
def save_dataset(X, y, path="data"):
    os.makedirs(path, exist_ok=True)

    np.save(os.path.join(path, "X.npy"), X)
    np.save(os.path.join(path, "y.npy"), y)

    print(f"Dataset salvo em: {path}")


# ================================
# Carregar dataset
# ================================
def load_dataset(path="data"):
    X = np.load(os.path.join(path, "X.npy"))
    y = np.load(os.path.join(path, "y.npy"))

    return X, y


# ================================
# Teste rápido (debug)
# ================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X, y = generate_dataset(
        n_samples=5,
        n_peaks=1,
        noise_std=0.02,
        seed=42
    )

    for i in range(5):
        plt.plot(X[i])
        plt.title(f"Params: {y[i]}")
        plt.show()