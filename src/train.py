import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.dataset import generate_dataset, normalize_dataset, save_dataset, load_dataset
from src.model import create_model


# ================================
# Configurações
# ================================
N_SAMPLES = 10000
N_PEAKS = 1
NOISE_STD = 0.02

TEST_SIZE = 0.2
VAL_SIZE = 0.1

EPOCHS = 50
BATCH_SIZE = 32

DATA_PATH = "data"
RESULTS_PATH = "results"


# ================================
# Preparar dataset
# ================================
def prepare_data(generate_new=True):
    if generate_new:
        print("Gerando dataset...")
        X, y = generate_dataset(
            n_samples=N_SAMPLES,
            n_peaks=N_PEAKS,
            noise_std=NOISE_STD,
            seed=42
        )

        X = normalize_dataset(X)
        save_dataset(X, y, DATA_PATH)

    else:
        print("Carregando dataset...")
        X, y = load_dataset(DATA_PATH)

    return X, y


# ================================
# Dividir dataset
# ================================
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE + VAL_SIZE, random_state=42
    )

    val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ================================
# Treinar modelo
# ================================
def train_model(X_train, y_train, X_val, y_val):
    model = create_model(
        input_size=X_train.shape[1],
        n_outputs=y_train.shape[1]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    return model, history


# ================================
# Salvar modelo e resultados
# ================================
def save_results(model, history):
    os.makedirs(RESULTS_PATH, exist_ok=True)

    model.save(os.path.join(RESULTS_PATH, "model.h5"))

    # Salvar gráfico de loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'val'])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(RESULTS_PATH, "loss.png"))
    plt.close()

    print("Resultados salvos em:", RESULTS_PATH)


# ================================
# Avaliação básica
# ================================
def evaluate_model(model, X_test, y_test):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test MAE: {mae:.4f}")

    preds = model.predict(X_test)

    print("\nExemplos de predição:")
    for i in range(5):
        print(f"Real: {y_test[i]} | Pred: {preds[i]}")


# ================================
# Pipeline principal
# ================================
def train(generate_new_data=True):
    # 1. Dados
    X, y = prepare_data(generate_new_data)

    # 2. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 3. Treino
    model, history = train_model(X_train, y_train, X_val, y_val)

    # 4. Avaliação
    evaluate_model(model, X_test, y_test)

    # 5. Salvar
    save_results(model, history)

    return model, history


# ================================
# Execução direta
# ================================
if __name__ == "__main__":
    train(generate_new_data=True)