"""Compare candidate ML algorithms using 5-fold cross-validation ROC-AUC.

This file is kept separate so the model-selection step is easy to show during
project review. It uses the same preprocessing and candidate models as the
main training pipeline and does not use the test set for model selection.
"""
from ml.pipeline import acquire, prepare, select_model
from ml.config import DATASETS


def compare_models():
    for key in ("heart", "kidney", "eye"):
        print("\n" + "=" * 72)
        print(f"MODEL COMPARISON: {DATASETS[key]['title']}")
        print("=" * 72)

        raw, _ = acquire(key)
        Xtr, Xte, ytr, yte, pre, _ = prepare(key, raw)
        selected_name, _, table = select_model(pre, Xtr, ytr)

        print("\nAlgorithm                    CV ROC-AUC    Selected")
        print("-" * 55)
        for row in table:
            mark = "YES" if row["selected"] else ""
            print(f"{row['model']:<28} {row['roc_auc']:<12.4f} {mark}")

        print(f"\nSelected model: {selected_name}")
        print("Selection rule: highest mean 5-fold stratified CV ROC-AUC.")
        print("The held-out test set is used only after selection for final evaluation.")


if __name__ == "__main__":
    compare_models()
