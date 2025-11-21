import numpy as np
import matplotlib.pyplot as plt

# X-axis: number of bags
x = np.array([1, 3, 5, 7])

# Example mean accuracies (replace with your values)
mean_acc = [
    [0.362, 0.338, 0.4578, 0.4122],  # method 1
    [0.4564, 0.6148, 0.6202, 0.7436],  # method 2
    [0.1702, 0.1112, 0.2174, 0.219]   # method 3
]

# Example std deviations (replace with your values)
std_acc = [
    [0.07228416147,  0.06619667665,   0.058593515, 0.1158088943],   # method 1
    [0.002701851217, 0.006058052492,  0.03336465195,   0.02233383084],  # method 2
    [0.08588480657,  0.04351091817,   0.05665509686,   0]    # method 3
]

import matplotlib.pyplot as plt

labels = ["LLP-KL", "Supervised GNN", "LLP-PCE"]
colors = ["tab:blue", "tab:orange", "tab:green"]

plt.figure(figsize=(8, 6))
for m in range(3):
    plt.errorbar(
        x, mean_acc[m], yerr=std_acc[m],
        label=labels[m], color=colors[m],
        marker="o", capsize=5, linestyle="-"
    )

# Increase font sizes
plt.xlabel("Number of Bags", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.title("Accuracy vs Number of Bags", fontsize=18)

# Make tick labels larger
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Larger legend font
plt.legend(fontsize=18)

# Optional: thicker grid lines
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
