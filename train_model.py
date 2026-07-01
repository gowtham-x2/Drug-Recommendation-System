from drug_system import METRICS_PATH, MODEL_PATH, train_and_save


def main() -> None:
    _, metrics = train_and_save()
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
