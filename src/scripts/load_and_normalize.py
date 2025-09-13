import pandas as pd
from pathlib import Path
from opera import load_merge, load_and_normal


def run_and_normalize():
    out = Path(__file__).resolve().parents[2] / 'data' / 'processed'
    df = load_merge()

    df = load_and_normal(df)

    df.to_csv(out / "processed_samples.csv", index=False)
    print("wrote processed samples to data")
    return df

if __name__ == "__main__":
    run_and_normalize()
