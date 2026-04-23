import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# =========================
# CARREGAR DADOS
# =========================
X = np.load("X.npy")
y_E = np.load("y_E.npy")  # já normalizado

# =========================
# SPLIT TREINO / TESTE
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_E, test_size=0.2, random_state=42
)

# =========================
# MODELO 
# =========================
model = Sequential([
    Dense(128, activation='relu', input_shape=(64,)),
    Dropout(0.2),

    Dense(128, activation='relu'),
    Dropout(0.2),

    Dense(64, activation='relu'),

    Dense(32, activation='relu'),

    Dense(4, activation='linear')  # 4 energias
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# =========================
# TREINAMENTO
# =========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,
    batch_size=32
)

# =========================
# SALVAR MODELO
# =========================
model.save("model.h5")

print("Modelo treinado com sucesso!")