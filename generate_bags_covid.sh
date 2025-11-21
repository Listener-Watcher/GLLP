#!/bin/bash

# Ensure the script runs from its own directory
cd "$(dirname "$0")"

datasets=("Covidbinary")
bagSizes=(1 3 5 7 10)
bagnums=(10 20 30 40 50 60 70 80)
for ds in "${datasets[@]}"; do
  for n in "${bagSizes[@]}"; do
    for k in "${bagnums[@]}"; do
      echo "=== Dataset: $ds | Bags: $n ==="

      python cora_bag_generator.py \
        --scheme random_iid \
        --iid-disjoint \
        --name "$ds" \
        --out-dir "outdisjoint$k/$ds/$n/iid/" \
        --num-bags "$n" \
        --bag-size "$k" \
        --k 2 \
        --fraction 0.5

      echo ""
    done
  done
done
