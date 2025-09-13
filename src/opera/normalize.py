import numpy as np

def load_and_normal(df):
    alpha = np.radians(df["alpha_deg"])

    Fx = df["Fx_N"]
    Fz = df["Fz_N"]

    My = df["My_Nm"]

    q = df["q_dyn_Pa"]

    c = df["c_ref_m"]
    S = df["S_ref_m2"]

    out = df.copy()

    L = Fx * np.sin(alpha) - Fz * np.cos(alpha)
    D = Fx * np.cos(alpha) + Fz * np.sin(alpha)
    out["D"] = D
    out["L"] = L
    out["CL"] = L / (q * S)
    out["CD"] = D / (q * S)
    out["CM"] = My / (q * S * c)
    return out

