"""Run the full ML pipeline for both diseases: python train.py"""
from ml.pipeline import run

if __name__ == "__main__":
    for k in ("heart", "kidney", "eye"):
        run(k)
    print("\nDone. Models saved to models/ and reports to reports/.")
