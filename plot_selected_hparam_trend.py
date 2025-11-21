# plot_selected_hparam_trend.py
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_subparams(s: str) -> Dict[str, Any]:
    out = {}
    if pd.isna(s):
        return out
    for part in str(s).split(","):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip()
            try:
                if re.match(r"^-?\d+(\.\d+)?$", v):
                    v = float(v) if "." in v else int(v)
            except Exception:
                pass
            out[k] = v
    return out


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    return None


def to_int_series(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    return s.round().astype("Int64")


def load_selected_runs(xlsx_path: str, method_sheet: str, use_complete_5seeds: bool = True) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=method_sheet)

    dataset_col = find_col(df, ["dataset", "data", "ds", "name"])
    bags_col    = find_col(df, ["bags", "bag", "num_bags", "b"])
    nodes_col   = find_col(df, ["nodes", "num_nodes"]) or find_col(df, ["n"])
    mean_col    = find_col(df, ["mean_final_test", "mean", "acc_mean", "final_mean"])
    std_col     = find_col(df, ["std_final_test", "std", "acc_std", "final_std"])
    subparams   = find_col(df, ["winning_subparams_mode", "subparams", "params"])
    complete    = find_col(df, ["complete_5seeds", "complete", "full_5seeds"])

    missing = [k for k, v in [
        ("dataset", dataset_col),
        ("bags", bags_col),
        ("nodes", nodes_col),
        ("mean", mean_col),
        ("std", std_col),
        ("winning_subparams_mode", subparams),
    ] if v is None]
    if missing:
        raise ValueError(f"Sheet '{method_sheet}' missing columns: {missing}")

    df = df.rename(columns={
        dataset_col: "dataset",
        bags_col: "bags_raw",
        nodes_col: "nodes_raw",
        mean_col: "mean",
        std_col: "std",
        subparams: "subparams",
    })

    if use_complete_5seeds and (complete is not None) and (complete in df.columns):
        df = df[df[complete] == True]  # noqa: E712

    df["bags"]  = to_int_series(df["bags_raw"])
    df["nodes"] = to_int_series(df["nodes_raw"])

    parsed = df["subparams"].apply(parse_subparams)
    df["beta"]     = parsed.apply(lambda d: d.get("beta", np.nan)).astype(float)
    df["graphreg"] = parsed.apply(lambda d: d.get("graphreg", np.nan)).astype(float)

    # Keep selected/best rows as-is (summary sheet already picked winners)
    return df[["dataset", "bags", "nodes", "mean", "std", "beta", "graphreg"]].copy()


def make_trend_table(
    df: pd.DataFrame,
    x_axis: str = "nodes",              # "nodes" or "bags"
    param: str = "beta",                # "beta" or "graphreg"
    agg_mode: str = "average",          # "average" or "fix"
    other_value: Optional[int] = None,  # if agg_mode="fix": value for the other dim
    weighted: bool = True,              # accuracy-weighted average of the selected param
) -> pd.DataFrame:
    assert x_axis in ("nodes", "bags")
    assert param in ("beta", "graphreg")
    assert agg_mode in ("average", "fix")
    other_dim = "bags" if x_axis == "nodes" else "nodes"

    d = df.copy()
    d = d[d[param].notna()]

    if agg_mode == "fix":
        if other_value is None:
            raise ValueError(f"When agg_mode='fix', provide other_value to fix {other_dim}.")
        d = d[d[other_dim] == other_value]

    keys = ["dataset", x_axis]

    if weighted:
        # accuracy-weighted average of the selected param
        d["w"]  = d["mean"].clip(lower=0)
        d["wp"] = d["w"] * d[param]
        agg = (
            d.groupby(keys, as_index=False)
             .agg(wp=("wp", "sum"), w=("w", "sum"))
        )
        agg["avg_param"] = np.where(agg["w"] > 0, agg["wp"] / agg["w"], np.nan)
        out = agg[["dataset", x_axis, "avg_param"]]
    else:
        # simple (unweighted) average
        out = (
            d.groupby(keys, as_index=False)[param]
             .mean()
             .rename(columns={param: "avg_param"})
        )

    out = out.rename(columns={x_axis: "x"})
    return out.sort_values(["dataset", "x"]).reset_index(drop=True)




def plot_trend_lines(
    tbl: pd.DataFrame,
    method: str,
    x_label: str,
    y_label: str,
    out_dir: str = "figs_selected_param",
    show: bool = False,
):
    """
    One figure per method with one line per dataset (clean & compact).
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for ds, g in tbl.groupby("dataset"):
        g = g.sort_values("x")
        plt.plot(g["x"], g["avg_param"], marker="o", label=str(ds))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{method}: {y_label} vs {x_label}")
    # plt.legend()
    # Make tick labels larger
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Larger legend font
    plt.legend(fontsize=14)
    plt.tight_layout()
    fname = f"{method}_{y_label.replace(' ', '')}_vs_{x_label.replace(' ', '')}.png"
    plt.savefig(outp / fname, dpi=200)
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # === CONFIG ===
    XLSX    = "methods_summary.xlsx"
    METHOD  = "GLLP"        # sheet name
    X_AXIS  = "nodes"          # "nodes" or "bags"
    PARAM   = "beta"           # "beta" or "graphreg"
    AGG     = "average"        # "average" or "fix"
    OTHER   = None             # e.g., if X_AXIS="nodes" and AGG="fix", set OTHER=5 to fix bags=5
    WEIGHT  = True             # accuracy-weighted average of the selected param

    # === RUN ===
    df = load_selected_runs(XLSX, METHOD, use_complete_5seeds=True)
    trend_tbl = make_trend_table(df, x_axis=X_AXIS, param=PARAM, agg_mode=AGG, other_value=OTHER, weighted=WEIGHT)
    x_label = "Number of nodes" if X_AXIS == "nodes" else "Number of bags"
    y_label = "Avg selected β" if PARAM == "beta" else r"Avg selected $\lambda$"
    plot_trend_lines(trend_tbl, METHOD, x_label, y_label, out_dir="figs_selected_param", show=False)
    print("✅ Saved figure to figs_selected_param/")
