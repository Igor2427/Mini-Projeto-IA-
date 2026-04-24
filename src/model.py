from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam


# ================================
# Criar modelo
# ================================
def create_model(
    input_size,
    n_outputs=2,
    learning_rate=0.001
):
    """
    Cria um modelo MLP para regressão.

    Parâmetros:
    - input_size: número de pontos da curva (ex: 100)
    - n_outputs: número de saídas (2 para 1 pico, 4 para 2 picos, etc.)
    """

    model = Sequential([
        Input(shape=(input_size,)),

        Dense(128, activation='relu'),
        BatchNormalization(),

        Dense(64, activation='relu'),
        BatchNormalization(),

        Dense(32, activation='relu'),

        Dense(n_outputs)  # saída linear
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    return model


# ================================
# Resumo do modelo
# ================================
def print_model_summary(model):
    model.summary()


# ================================
# Teste rápido
# ================================
if __name__ == "__main__":
    model = create_model(input_size=100, n_outputs=2)
    print_model_summary(model)