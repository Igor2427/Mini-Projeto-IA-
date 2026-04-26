import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

from src.dataset import load_dataset


# ================================
# Configurações
# ================================
DATA_PATH = "data"
MODEL_PATH = "results/model.h5"


# ================================
# Métricas
# ================================
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    return mae, mse


# ================================
# Avaliação global
# ================================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae, mse = compute_metrics(y_test, y_pred)

    print("===== RESULTADOS GLOBAIS =====")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")

    return y_pred


# ================================
# Scatter plots (real vs previsto)
# ================================
def plot_predictions(y_true, y_pred):
    n_outputs = y_true.shape[1]

    for i in range(n_outputs):
        plt.figure()
        plt.scatter(y_true[:, i], y_pred[:, i])
        plt.xlabel("Valor Real")
        plt.ylabel("Valor Previsto")
        plt.title(f"Saída {i} (Real vs Previsto)")

        # linha ideal
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.grid()
        plt.show()


# ================================
# Histograma de erros
# ================================
def plot_error_distribution(y_true, y_pred):
    errors = y_true - y_pred
    n_outputs = y_true.shape[1]

    for i in range(n_outputs):
        plt.figure()
        plt.hist(errors[:, i], bins=50)
        plt.title(f"Distribuição do erro - saída {i}")
        plt.xlabel("Erro")
        plt.ylabel("Frequência")
        plt.grid()
        plt.show()


# ================================
# Visualização de curvas
# ================================
def plot_sample_curves(X, y_true, y_pred, n_samples=5):
    for i in range(n_samples):
        plt.figure()
        plt.plot(X[i], label="Curva TDS")

        plt.title(
            f"Real: {np.round(y_true[i], 2)}\nPred: {np.round(y_pred[i], 2)}"
        )
        plt.legend()
        plt.grid()
        plt.show()


# ================================
# Pipeline principal
# ================================
def run_evaluation():
    print("Carregando dataset...")
    X, y = load_dataset(DATA_PATH)

    print("Carregando modelo...")
    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH, compile=False)

    # dividir (mesma lógica do treino)
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # avaliação
    y_pred = evaluate_model(model, X_test, y_test)

    # gráficos
    plot_predictions(y_test, y_pred)
    plot_error_distribution(y_test, y_pred)
    plot_sample_curves(X_test, y_test, y_pred)


# ================================
# Execução direta
# ================================
if __name__ == "__main__":
    run_evaluation()