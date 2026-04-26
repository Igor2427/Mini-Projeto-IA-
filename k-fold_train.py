import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold

from src.dataset import generate_dataset, normalize_dataset, save_dataset, load_dataset
from src.model import create_model


# ================================
# Configurações
# ================================
N_SAMPLES = 10000
N_PEAKS = 1
NOISE_STD = 0.02

TEST_SIZE = 0.2  # separado FINAL
K_FOLDS = 5

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
# K-Fold Training
# ================================
def train_kfold(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold = 1
    all_mae = []
    all_loss = []

    best_model = None
    best_loss = float("inf")

    for train_idx, val_idx in kf.split(X):
        print(f"\n===== FOLD {fold} =====")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = create_model(
            input_size=X.shape[1],
            n_outputs=y.shape[1]
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )

        loss, mae = model.evaluate(X_val, y_val, verbose=0)

        print(f"Fold {fold} - Loss: {loss:.4f} | MAE: {mae:.4f}")

        all_loss.append(loss)
        all_mae.append(mae)

        # salvar melhor modelo
        if loss < best_loss:
            best_loss = loss
            best_model = model
            best_history = history

        fold += 1

    print("\n===== RESULTADO MÉDIO =====")
    print(f"Loss médio: {np.mean(all_loss):.4f}")
    print(f"MAE médio: {np.mean(all_mae):.4f}")

    return best_model, best_history


# ================================
# Avaliação FINAL (teste)
# ================================
def evaluate_model(model, X_test, y_test):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)

    print("\n===== TESTE FINAL =====")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test MAE: {mae:.4f}")

    preds = model.predict(X_test)

    print("\nExemplos de predição:")
    for i in range(5):
        print(f"Real: {y_test[i]} | Pred: {preds[i]}")


# ================================
# Salvar resultados
# ================================
def save_results(model, history):
    os.makedirs(RESULTS_PATH, exist_ok=True)

    model.save(os.path.join(RESULTS_PATH, "model.h5"))

    # gráfico de loss
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
# Pipeline principal
# ================================
def train(generate_new_data=True):

    # 1. Dados
    X, y = prepare_data(generate_new_data)

    # 2. Separar TESTE FINAL (20%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

    # 3. K-Fold no treino
    model, history = train_kfold(X_train_full, y_train_full, k=K_FOLDS)

    # 4. Teste final (nunca visto pelo modelo)
    evaluate_model(model, X_test, y_test)

    # 5. Salvar
    save_results(model, history)

    return model, history


# ================================
# Execução direta
# ================================
if __name__ == "__main__":
    train(generate_new_data=True)