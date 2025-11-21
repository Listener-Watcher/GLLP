#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def aggregate_over(df, group_fields, x_field):
    """
    Aggregates df (which already contains mean_final_test and std_final_test over seeds)
    across the 'other' dimension by averaging mean_final_test and estimating dispersion
    between the grouped points.
    Returns a tidy DataFrame with columns: method, x_field, mean, std
    """
    out = []
    for (method, x), g in df.groupby(["method", x_field]):
        # Average the mean over the other dimension(s)
        mean_over = g["mean_final_test"].mean()
        # Use the std of those means across the other dimension(s) as error bars
        std_over = g["mean_final_test"].std(ddof=1)
        out.append({ "method": method, x_field: x, "mean": mean_over, "std": std_over })
    return pd.DataFrame(out).sort_values([x_field, "method"])

def plot_lines_with_errorbars(df_ax, x_field, title, outpath):
    plt.figure(figsize=(8,5))
    methods = sorted(df_ax["method"].unique())
    for m in methods:
        sub = df_ax[df_ax["method"] == m].sort_values(x_field)
        # Use errorbar with caps
        plt.errorbar(sub[x_field], sub["mean"], yerr=sub["std"], marker='o', linestyle='-', capsize=3, label=m)
    plt.xlabel("Number of " + ("Bags" if x_field=="bags" else "Nodes"))
    plt.ylabel("Final Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="mean_over_seeds_by_method.csv", help="Path to mean_over_seeds_by_method.csv")
    ap.add_argument("--dataset", default="Cora", help="Dataset to plot (e.g., Cora, CiteSeer, PubMed, Computers, Photos)")
    ap.add_argument("--outdir", default=None, help="Directory to write PNGs (default: same dir as CSV)")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}. Run your aggregation first to produce this file.")

    df = pd.read_csv(args.csv)
    # Expect columns: dataset, bags, nodes, method, mean_final_test, std_final_test, n, winning_subparams_mode, ...
    needed = {"dataset","bags","nodes","method","mean_final_test","std_final_test"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df_ds = df[df["dataset"] == args.dataset].copy()
    if df_ds.empty:
        raise ValueError(f"No rows for dataset={args.dataset}. Available: {sorted(df['dataset'].unique())}")

    # Where to save
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.csv)) or "."
    os.makedirs(outdir, exist_ok=True)

    # 1) X = bags (aggregate across nodes)
    df_bags = aggregate_over(df_ds, group_fields=["method","bags"], x_field="bags")
    out_bags = os.path.join(outdir, f"{args.dataset}_mean_vs_bags.png")
    plot_lines_with_errorbars(df_bags, "bags", f"{args.dataset}: Accuracy vs Number of Bags (mean±std across nodes)", out_bags)

    # 2) X = nodes (aggregate across bags)
    df_nodes = aggregate_over(df_ds, group_fields=["method","nodes"], x_field="nodes")
    out_nodes = os.path.join(outdir, f"{args.dataset}_mean_vs_nodes.png")
    plot_lines_with_errorbars(df_nodes, "nodes", f"{args.dataset}: Accuracy vs Number of Nodes (mean±std across bags)", out_nodes)

    print("Wrote:")
    print(" ", out_bags)
    print(" ", out_nodes)

if __name__ == "__main__":
    main()
