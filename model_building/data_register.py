"""Validate that the raw tourism dataset has the expected schema."""
import os
import sys
import pandas as pd

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

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

def main() -> int:
    print(f"REPO_ROOT : {REPO_ROOT}")
    print(f"RAW_PATH  : {RAW_PATH}")

    if not os.path.exists(RAW_PATH):
        print(f"[FAIL] Dataset not found at {RAW_PATH}", file=sys.stderr)
        return 1

    df = pd.read_csv(RAW_PATH)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[FAIL] Missing columns: {missing}", file=sys.stderr)
        return 1

    if df["ProdTaken"].isna().any():
        print("[FAIL] ProdTaken has nulls.", file=sys.stderr)
        return 1
    if not set(df["ProdTaken"].unique()).issubset({0, 1}):
        print("[FAIL] ProdTaken must be binary (0/1).", file=sys.stderr)
        return 1

    print("Dataset registered successfully.")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("ProdTaken distribution:")
    print(df["ProdTaken"].value_counts(normalize=True).round(4))
    return 0

if __name__ == "__main__":
    sys.exit(main())
