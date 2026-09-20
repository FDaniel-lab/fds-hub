"""Split the dataset into train/test and persist the CSVs."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def find_project_root(marker="data/tourism.csv", start=None):
    start = os.path.abspath(start or os.path.dirname(__file__))
    current = start
    for _ in range(10):
        if os.path.exists(os.path.join(current, marker)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise FileNotFoundError(f"Could not locate {marker} from {start}")

REPO_ROOT = find_project_root()
RAW_PATH  = os.path.join(REPO_ROOT, "data", "tourism.csv")

OUT_DIR = os.environ.get("SPLIT_OUTPUT_DIR", os.getcwd())
os.makedirs(OUT_DIR, exist_ok=True)

TARGET    = "ProdTaken"
DROP_COLS = ["CustomerID"]

def main():
    print(f"Reading from: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH).drop(columns=DROP_COLS)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    Xtrain.to_csv(os.path.join(OUT_DIR, "Xtrain.csv"), index=False)
    Xtest.to_csv( os.path.join(OUT_DIR, "Xtest.csv"),  index=False)
    ytrain.to_csv(os.path.join(OUT_DIR, "ytrain.csv"), index=False)
    ytest.to_csv( os.path.join(OUT_DIR, "ytest.csv"),  index=False)

    print(f"Splits written to {OUT_DIR}")
    print("Train ProdTaken rate:", round(ytrain.mean(), 4))
    print("Test  ProdTaken rate:", round(ytest.mean(), 4))

if __name__ == "__main__":
    main()
