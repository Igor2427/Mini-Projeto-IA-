import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# =========================
# CARREGAR DADOS
# =========================
X = np.load("X.npy")
y_E = np.load("y_E.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_E, test_size=0.2, random_state=42
)

# =========================
# CARREGAR MODELO
# =========================
model = load_model("model.h5", compile=False)

# =========================
# PREDIÇÃO
# =========================
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, 1)

# =========================
# DESNORMALIZAÇÃO
# =========================
y_test_real = y_test * 120000
y_pred_real = y_pred * 120000

# =========================
# MÉTRICAS
# =========================
mse = mean_squared_error(y_test_real, y_pred_real)
mae = mean_absolute_error(y_test_real, y_pred_real)

print("MSE:", mse)
print("MAE:", mae)

# =========================
# GRÁFICO
# =========================
plt.figure(figsize=(8,6))

plt.scatter(y_test_real.flatten(), y_pred_real.flatten(), alpha=0.5)
plt.plot(
    [y_test_real.min(), y_test_real.max()],
    [y_test_real.min(), y_test_real.max()],
    'r--'
)

plt.xlabel("Valor Real (Energia)")
plt.ylabel("Valor Predito")
plt.title("Comparação: Real vs Predito")

plt.grid()
plt.show()