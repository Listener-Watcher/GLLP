# GLLP
Graph Label Distribution Learning   
# environment Setup   
See requirement.txt from pip and enviornment.yaml from conda. We mainly use torch_goemtric for graph learning and POT for Optimal transport. Please adjust torch_geoemtric version for suitable cuda version.   
# Dataset Generation
See generate_bags.sh file for general dataset. For Covid dataset, please use generate_bags_covid.sh. The raw data is provided in data folder.

# Train
See run_exp_5_seeds.sh file and others. We provide bash files for running under linux environment. For windows user, please convert it to .ps1 file.

```
run_exp_5_seeds.sh > results.txt 2>&1
```
# parsing and compute.
Use parse_and_compute.py
```
parse_and_compute.py results.txt
```
# plot
Use plot_mean_over_seeds.py
```
plot_mean_over_seeds.py --dataset Cora --outdir figure/
```
