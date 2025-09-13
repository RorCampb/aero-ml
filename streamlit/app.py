import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os


st.set_page_config(page_title="Aero Polars", layout="wide")

PROJ_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJ_ROOT / "data" / "processed" / "processed_samples.csv"

# Cache the CSV but bust cache when file changes
def file_mtime(p: Path) -> float:
    try:
        return os.path.getmtime(p)
    except FileNotFoundError:
        return 0.0

@st.cache_data(show_spinner=False)
def load_df(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)

mtime = file_mtime(CSV_PATH)
if mtime == 0.0:
    st.error(f"CSV not found: {CSV_PATH}")
    st.stop()

# Load once (cached). You don't need session_state for the df unless you plan to mutate it.
df = load_df(str(CSV_PATH), mtime)

st.title("Aero Polars")

run_id = st.selectbox("Run", sorted(df["run_id"].unique()))
g = df[df["run_id"] == run_id].sort_values("alpha_deg")

# CL vs alpha with fitted line in linear region
fig1, ax1 = plt.subplots()
ax1.scatter(g["alpha_deg"], g["CL"], s=20, label="data")

mask = (g["alpha_deg"] >= -2) & (g["alpha_deg"] <= 6)
if mask.sum() >= 2:
    m_per_deg, b = np.polyfit(g.loc[mask, "alpha_deg"], g.loc[mask, "CL"], 1)
    xx = np.linspace(-2, 6, 100)
    ax1.plot(xx, m_per_deg * xx + b, linestyle="--",
             label=f"slope {m_per_deg:.3f} /deg  (≈ {m_per_deg*180/np.pi:.2f} /rad)")

ax1.set_xlabel("α [deg]")
ax1.set_ylabel("CL")
ax1.grid(True)
ax1.legend(loc="best")
st.pyplot(fig1)

# Optional: CD vs CL plot too
fig2, ax2 = plt.subplots()
ax2.scatter(g["CL"], g["CD"], s=20)
ax2.set_xlabel("CL")
ax2.set_ylabel("CD")
ax2.grid(True)
st.pyplot(fig2)

st.caption(f"Loaded {CSV_PATH} (mtime: {int(mtime)})")

