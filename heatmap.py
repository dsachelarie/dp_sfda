import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

acc = [[56.5, 64.0, 64.0, 65.7, 75.7], [63.9, 80.5, 81.3, 82.6, 90.8], [66.0, 82.9, 83.2, 84.9, 91.2], [69.3, 84.1, 84.6, 85.3, 91.7], [72.2, 79.4, 85.0, 86.5, 97.9]]
annot = np.array([[f"{v:.1f}" for v in row] for row in acc])
annot[4, 4] = ""

acc_few = [[63.9, 69.6, 69.4, 70.4, 78.9], [72.1, 81.8, 83.0, 85.0, 91.6], [75.0, 84.3, 86.6, 87.4, 91.9], [77.4, 85.7, 86.6, 87.9, 92.5], [84.6, 87.5, 88.4, 88.1, 97.9]]
annot_few = np.array([[f"{v:.1f}" for v in row] for row in acc_few])
annot_few[4, 4] = ""

acc2 = [[21.1, 67.5, 70.5, 66.5, 90.5], [21.9, 77.0, 78.8, 79.9, 95.4], [22.4, 78.3, 80.5, 85.0, 96.0], [22.9, 78.9, 80.8, 85.4, 96.4], [23.2, 74.6, 77.8, 86.9, 98.0]]
annot2 = np.array([[f"{v:.1f}" for v in row] for row in acc2])
annot2[4, 4] = ""

acc2_few = [[48.6, 72.7, 75.4, 75.9, 92.1], [61.0, 80.8, 84.8, 84.5, 96.6], [64.9, 82.2, 85.9, 85.8, 97.1], [68.6, 81.5, 85.6, 85.1, 97.3], [68.8, 73.2, 76.9, 76.4, 98.0]]
annot2_few = np.array([[f"{v:.1f}" for v in row] for row in acc2_few])
annot2_few[4, 4] = ""

acc3 = [[26.2, 64.9, 66.6, 67.2, 90.9], [27.4, 71.7, 73.4, 74.0, 95.9], [27.7, 72.9, 74.2, 74.9, 96.7], [27.8, 72.8, 75.5, 76.0, 96.7], [27.5, 70.9, 77.4, 77.3, 98.9]]
annot3 = np.array([[f"{v:.1f}" for v in row] for row in acc3])
annot3[4, 4] = ""

acc3_few = [[60.9, 84.1, 85.2, 86.3, 94.0], [71.4, 93.9, 93.2, 94.6, 96.4], [74.4, 94.3, 93.4, 94.5, 96.5], [77.4, 93.3, 93.4, 95.0, 96.6], [76.4, 91.4, 90.9, 91.2, 98.9]]
annot3_few = np.array([[f"{v:.1f}" for v in row] for row in acc3_few])
annot3_few[4, 4] = ""

vmin = min(min(min(d)) for d in [acc, acc2, acc3, acc_few, acc2_few, acc3_few])
vmax = max(max(max(d)) for d in [acc, acc2, acc3, acc_few, acc2_few, acc3_few])
labels = ['0.1', '1', '4', '8', '∞']

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
cbar_ax = fig.add_axes([0.93, 0.3, 0.015, 0.4])

ax = sns.heatmap(acc, ax=axes[0], cmap='viridis', annot=annot, fmt="", xticklabels=labels, yticklabels=labels, vmin=vmin, vmax=vmax, cbar=False)
ax.text(4 + 0.5, 4 + 0.5, "97.9*", ha='center', va='center', color='black', fontsize=10)
ax.set_title("DP-SHOT M→U")
ax.set_xlabel("Source ε")
ax.set_ylabel("Target ε")
ax.invert_yaxis()

ax = sns.heatmap(acc_few, ax=axes[1], cmap='viridis', annot=annot_few, fmt="", xticklabels=labels, yticklabels=labels, vmin=vmin, vmax=vmax, cbar=True, cbar_ax=cbar_ax)
ax.text(4 + 0.5, 4 + 0.5, "97.9*", ha='center', va='center', color='black', fontsize=10)
ax.set_title("DP-FewSHOT M→U")
ax.set_xlabel("Source ε")
ax.set_ylabel("Target ε")
ax.invert_yaxis()
plt.savefig("heatmap_m_u.png", dpi=300, bbox_inches='tight')

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
cbar_ax = fig.add_axes([0.93, 0.3, 0.015, 0.4])

ax = sns.heatmap(acc2, ax=axes[0], cmap='viridis', annot=annot2, fmt="", xticklabels=labels, yticklabels=labels, vmin=vmin, vmax=vmax, cbar=False)
ax.text(4 + 0.5, 4 + 0.5, "98.0*", ha='center', va='center', color='black', fontsize=10)
ax.set_title("DP-SHOT U→M")
ax.set_xlabel("Source ε")
ax.set_ylabel("Target ε")
ax.invert_yaxis()

ax = sns.heatmap(acc2_few, ax=axes[1], cmap='viridis', annot=annot2_few, fmt="", xticklabels=labels, yticklabels=labels, vmin=vmin, vmax=vmax, cbar=True, cbar_ax=cbar_ax)
ax.text(4 + 0.5, 4 + 0.5, "98.0*", ha='center', va='center', color='black', fontsize=10)
ax.set_title("DP FewSHOT U→M")
ax.set_xlabel("Source ε")
ax.set_ylabel("Target ε")
ax.invert_yaxis()
plt.savefig("heatmap_u_m.png", dpi=300, bbox_inches='tight')

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
cbar_ax = fig.add_axes([0.93, 0.3, 0.015, 0.4])

ax = sns.heatmap(acc3, ax=axes[0], cmap='viridis', annot=annot3, fmt="", xticklabels=labels, yticklabels=labels, vmin=vmin, vmax=vmax, cbar=False)
ax.text(4 + 0.5, 4 + 0.5, "98.9*", ha='center', va='center', color='black', fontsize=10)
ax.set_title("DP-SHOT S→M")
ax.set_xlabel("Source ε")
ax.set_ylabel("Target ε")
ax.invert_yaxis()

ax = sns.heatmap(acc3_few, ax=axes[1], cmap='viridis', annot=annot3_few, fmt="", xticklabels=labels, yticklabels=labels, vmin=vmin, vmax=vmax, cbar=True, cbar_ax=cbar_ax)
ax.text(4 + 0.5, 4 + 0.5, "98.9*", ha='center', va='center', color='black', fontsize=10)
ax.set_title("DP-FewSHOT S→M")
ax.set_xlabel("Source ε")
ax.set_ylabel("Target ε")
ax.invert_yaxis()
plt.savefig("heatmap_s_m.png", dpi=300, bbox_inches='tight')
