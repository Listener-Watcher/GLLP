import re
import sys
import pandas as pd
from collections import defaultdict, Counter

logfile = sys.argv[1] if len(sys.argv) > 1 else "./covid_output.txt"

# --- Patterns ---
pat_head = re.compile(r"^=== Dataset:\s*(.+?) \| Bags:\s*(\d+) \| Nodes:\s*(\d+) \| Seed:\s*(\d+)\s*===")
pat_val  = re.compile(r"final test=(\d+(?:\.\d+)?)")

# --- Expected order per (ds,bags,nodes,seed) block ---
# 1:  kl
# 2-4: klreg (k_aug = 1,5,10)  -> take max across these for KLREG
# 5:  klgraph (graphreg=0.5)
# 6-17: otkl grid over graphreg ∈ {0.5,1}, beta ∈ {0.05,0.5,1}, ep ∈ {10,30} -> take max across entire grid for OTKL
# 18: pt

kaug_list      = [1, 5, 10]                   # KLREG
graphreg_grid  = [0.5, 1.0]                   # OTKL
beta_grid      = [0.05, 0.5, 1.0]             # OTKL
ep_grid        = [10, 30]                     # OTKL

# index (0-based) -> label (method or subconfig) just for diagnostics
index_labels = []
index_labels.append(("kl", {}))  # 0
for ka in kaug_list:             # 1-3
    index_labels.append(("klreg", {"k_aug": ka}))
index_labels.append(("klgraph", {"graphreg": 0.5}))  # 4
for gr in graphreg_grid:         # 5-16
    for b in beta_grid:
        for ep in ep_grid:
            index_labels.append(("otkl", {"graphreg": gr, "beta": b, "ep": ep}))
index_labels.append(("pt", {}))  # 17
index_labels.append(("vae", {}))  # 17

EXPECTED_PER_SEED = len(index_labels)  # should be 18

# group membership by indices
grp_indices = {
    "kl":      [0],
    "klreg":   [1,2,3],
    "klgraph": [4],
    "otkl":    list(range(5, 17)),  # 5..16 inclusive
    "pt":      [17],
    "vae":     [18],
}

# --- Parse log ---
current_key = None
vals_buffer = []
rows_raw = []  # (ds,bags,nodes,seed, idx, method, subparams, value)

def flush_block(key, vals):
    if not vals:
        return
    ds, bags, nodes, seed = key
    # align/truncate if count differs
    n = min(len(vals), EXPECTED_PER_SEED)
    for idx in range(n):
        method, params = index_labels[idx]
        rows_raw.append({
            "dataset": ds,
            "bags": int(bags),
            "nodes": int(nodes),
            "seed": int(seed),
            "idx": idx,
            "method": method,
            "subparams": params,  # dict; we’ll stringify later
            "final_test": float(vals[idx]),
        })
    if len(vals) != EXPECTED_PER_SEED:
        print(f"[WARN] {key} produced {len(vals)} results (expected {EXPECTED_PER_SEED}). Parsed first {n}.")

with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        s = line.strip()
        m = pat_head.match(s)
        if m:
            if current_key is not None:
                flush_block(current_key, vals_buffer)
            ds, bags, nodes, seed = m.groups()
            current_key = (ds, int(bags), int(nodes), int(seed))
            vals_buffer = []
            continue
        m = pat_val.search(s)
        if m and current_key is not None:
            vals_buffer.append(float(m.group(1)))

if current_key is not None:
    flush_block(current_key, vals_buffer)

if not rows_raw:
    raise SystemExit("No results parsed. Check your log format.")

# --- DataFrame of raw entries ---
df_raw = pd.DataFrame(rows_raw)
# normalize subparams to a stable string for later reporting
def params_str(p):
    if not p: return ""
    keys = ["k_aug","graphreg","beta","ep"]
    return ",".join(f"{k}={p[k]}" for k in keys if k in p)
df_raw["subparams_str"] = df_raw["subparams"].apply(params_str)

# --- Reduce to "best within method group" PER SEED ---
best_rows = []
for (ds,bags,nodes,seed), g in df_raw.groupby(["dataset","bags","nodes","seed"]):
    # Helper to pick best by indices
    def best_of(method_name):
        idxs = grp_indices[method_name]
        subset = g[g["idx"].isin(idxs)]
        if subset.empty:
            return None
        # choose the row with max final_test; tie-break by preferring later row (stable but arbitrary)
        j = subset["final_test"].idxmax()
        row = subset.loc[j]
        return {
            "dataset": ds, "bags": bags, "nodes": nodes, "seed": seed,
            "method": method_name,
            "final_test": row["final_test"],
            "winning_subparams": row["subparams_str"],
        }

    for m in ["kl","klreg","klgraph","otkl","pt","vae"]:
        r = best_of(m)
        if r is not None:
            best_rows.append(r)

df_best = pd.DataFrame(best_rows)
df_best.to_csv("best_per_seed_long.csv", index=False)

# --- Average over seeds PER (dataset,bags,nodes,method) ---
# also record the most frequent winning_subparams (and among ties, the one with highest mean)
agg = (
    df_best.groupby(["dataset","bags","nodes","method"], as_index=False)
           .agg(mean_final_test=("final_test","mean"),
                std_final_test=("final_test","std"),
                n=("final_test","count"))
)

# Determine the most frequent winning subparams for each group
winner_param_rows = []
for key, group in df_best.groupby(["dataset","bags","nodes","method"]):
    # frequency of winners
    counts = Counter(group["winning_subparams"])
    # tie-breaker by higher average score
    best_param, best_count, best_mean = None, -1, -1.0
    for param, cnt in counts.items():
        mean_param = group.loc[group["winning_subparams"]==param, "final_test"].mean()
        if cnt > best_count or (cnt == best_count and mean_param > best_mean):
            best_param, best_count, best_mean = param, cnt, mean_param
    winner_param_rows.append((*key, best_param, best_count, best_mean))

df_winner = pd.DataFrame(winner_param_rows, columns=["dataset","bags","nodes","method","winning_subparams_mode","mode_count","mode_mean"])

# Merge winner info into agg
out = pd.merge(agg, df_winner, on=["dataset","bags","nodes","method"], how="left")
out = out.sort_values(["dataset","bags","nodes","method"])
out.to_csv("mean_over_seeds_by_method.csv", index=False)

print("Wrote:")
print("  - best_per_seed_long.csv        (best subconfig per seed within each method)")
print("  - mean_over_seeds_by_method.csv (averaged over seeds per method)")
