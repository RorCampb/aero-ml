import pandas as pd
from pathlib import Path


def load_merge():
    source = Path(__file__).resolve().parents[2] / 'data' / 'source'

    samples = pd.read_csv(source / "samples.csv")
    runs = pd.read_csv(source / "runs.csv")
    geometry = pd.read_csv(source / "geometry.csv")

    samples = samples.merge(
        runs[["run_id", "article_id"]],
        on="run_id",
        how="left"
    )

    df = samples.merge(
        geometry,
        on="article_id",
        how="left"
    )

    return df
