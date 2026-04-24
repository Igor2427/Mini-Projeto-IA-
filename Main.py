import argparse

from train import train
from evaluate import run_evaluation


# ================================
# Parser de argumentos
# ================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Projeto TDS com Redes Neurais"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "full"],
        help="Modo de execução: train | eval | full"
    )

    parser.add_argument(
        "--generate_data",
        type=bool,
        default=True,
        help="Gerar novo dataset (True/False)"
    )

    return parser.parse_args()


# ================================
# Main
# ================================
def main():
    args = parse_args()

    print("=== Projeto TDS ML ===")
    print(f"Modo: {args.mode}")

    # ----------------------------
    # MODO TREINO
    # ----------------------------
    if args.mode == "train":
        print("\n[1] Treinando modelo...")
        train(generate_new_data=args.generate_data)

    # ----------------------------
    # MODO AVALIAÇÃO
    # ----------------------------
    elif args.mode == "eval":
        print("\n[2] Avaliando modelo...")
        run_evaluation()

    # ----------------------------
    # MODO COMPLETO
    # ----------------------------
    elif args.mode == "full":
        print("\n[1] Treinando modelo...")
        train(generate_new_data=args.generate_data)

        print("\n[2] Avaliando modelo...")
        run_evaluation()

    print("\n=== Execução finalizada ===")


# ================================
# Execução direta
# ================================
if __name__ == "__main__":
    main()