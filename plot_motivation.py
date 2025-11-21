#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_vs(df, x_field, out_path, dataset=None, method=None):
    """
    Plot accuracy vs x_field (bags or nodes).
    If method is given, plot only that method.
    """
    plt.figure(figsize=(8,5))

    methods = [method] if method else df['method'].unique()
    for m in methods:
        sub = df[df['method'] == m]
        if dataset:
            sub = sub[sub['dataset'] == dataset]
        if sub.empty:
            continue

        grouped = sub.groupby([x_field, "dataset"], as_index=False).agg(
            mean=("mean_final_test", "mean"),
            std=("mean_final_test", "std")
        )
        plt.errorbar(grouped[x_field], grouped["mean"],
                     yerr=grouped["std"], marker='o', capsize=3,
                     label=m)

    plt.xlabel("Number of " + ("Bags" if x_field=="bags" else "Nodes"))
    plt.ylabel("Final Accuracy")
    title = f"Accuracy vs {x_field.capitalize()}"
    if dataset:
        title += f" ({dataset})"
    if method:
        title += f" [{method}]"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Wrote {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to methods_summary.xlsx")
    ap.add_argument("--dataset", default=None, help="Dataset to filter")
    ap.add_argument("--method", default=None, help="Method to plot (default: all methods)")
    ap.add_argument("--outdir", default="plots", help="Directory to save plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    dfs = []
    xls = pd.ExcelFile(args.xlsx)
    for method in xls.sheet_names:
        df = pd.read_excel(args.xlsx, sheet_name=method)
        df["method"] = method
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    # Plot vs bags
    out_bags = os.path.join(
        args.outdir,
        f"acc_vs_bags{'_'+args.dataset if args.dataset else ''}{'_'+args.method if args.method else ''}.png"
    )
    plot_vs(df_all, "bags", out_bags, dataset=args.dataset, method=args.method)

    # Plot vs nodes
    out_nodes = os.path.join(
        args.outdir,
        f"acc_vs_nodes{'_'+args.dataset if args.dataset else ''}{'_'+args.method if args.method else ''}.png"
    )
    plot_vs(df_all, "nodes", out_nodes, dataset=args.dataset, method=args.method)

if __name__ == "__main__":
    main()
