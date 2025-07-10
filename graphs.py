import numpy as np
import matplotlib.pyplot as plt

# Accuracy data
acc = np.array([[56.5, 64.0, 64.0, 65.7, 75.7], [63.9, 80.5, 81.3, 82.6, 90.8], [66.0, 82.9, 83.2, 84.9, 91.2], [69.3, 84.1, 84.6, 85.3, 91.7], [72.2, 79.4, 85.0, 86.5, 97.9]])
std = np.array([[8.6, 8.8, 6.3, 6.2, 5.9], [7.8, 7.3, 6.3, 6.6, 5.6], [8.7, 7.0, 6.2, 6.7, 6.6], [9.5, 7.2, 6.3, 6.8, 6.5], [11.5, 10.2, 6.3, 4.8, 0.2]])

acc_few = np.array([[63.9, 69.6, 69.4, 70.4, 78.9], [72.1, 81.8, 83.0, 85.0, 91.6], [75.0, 84.3, 86.6, 87.4, 91.9], [77.4, 85.7, 86.6, 87.9, 92.5], [84.6, 87.5, 88.4, 88.1, 97.9]])
std_few = np.array([[6.9, 6.8, 6.3, 4.4, 4.1], [6.8, 7.0, 6.5, 5.7, 4.4], [6.3, 6.7, 6.1, 5.7, 5.2], [6.7, 6.7, 6.3, 5.9, 4.8], [4.0, 1.3, 1.5, 1.2, 0.2]])

acc2 = np.array([[21.1, 67.5, 70.5, 66.5, 90.5], [21.9, 77.0, 78.8, 79.9, 95.4], [22.4, 78.3, 80.5, 85.0, 96.0], [22.9, 78.9, 80.8, 85.4, 96.4], [23.2, 74.6, 77.8, 86.9, 98.0]])
std2 = np.array([[7.7, 3.2, 2.1, 5.9, 0.5], [8.8, 3.8, 2.8, 3.3, 1.7], [8.9, 4.2, 3.5, 6.9, 1.6], [9.1, 5.4, 4.0, 6.6, 1.3], [9.3, 4.9, 2.9, 4.2, 0.6]])

acc2_few = np.array([[48.6, 72.7, 75.4, 75.9, 92.1], [61.0, 80.8, 84.8, 84.5, 96.6], [64.9, 82.2, 85.9, 85.8, 97.1], [68.6, 81.5, 85.6, 85.1, 97.3], [68.8, 73.2, 76.9, 76.4, 98.0]])
std2_few = np.array([[3.4, 3.6, 2.7, 2.6, 0.6], [4.2, 3.3, 2.8, 3.7, 0.3], [4.4, 3.7, 3.0, 3.8, 0.4], [5.2, 4.7, 3.3, 4.1, 0.1], [5.7, 4.1, 4.4, 4.6, 0.6]])

acc3 = np.array([[26.2, 64.9, 66.6, 67.2, 90.9], [27.4, 71.7, 73.4, 74.0, 95.9], [27.7, 72.9, 74.2, 74.9, 96.7], [27.8, 72.8, 75.5, 76.0, 96.7], [27.5, 70.9, 77.4, 77.3, 98.9]])
std3 = np.array([[10.8, 4.4, 4.3, 3.7, 2.9], [10.8, 3.1, 2.9, 2.6, 2.5], [11.3, 4.0, 3.1, 2.6, 2.3], [12.0, 4.2, 2.8, 3.1, 2.3], [10.2, 6.3, 5.0, 5.0, 0.1]])

acc3_few = np.array([[60.9, 84.1, 85.2, 86.3, 94.0], [71.4, 93.9, 93.2, 94.6, 96.4], [74.4, 94.3, 93.4, 94.5, 96.5], [77.4, 93.3, 93.4, 95.0, 96.6], [76.4, 91.4, 90.9, 91.2, 98.9]])
std3_few = np.array([[4.5, 2.7, 2.4, 2.3, 1.2], [5.2, 2.4, 3.2, 2.9, 2.3], [5.9, 3.4, 3.7, 3.8, 2.8], [6.4, 4.1, 4.3, 3.9, 2.9], [6.5, 2.9, 2.6, 3.4, 0.1]])

labels = ['0.1', '1', '4', '8', '∞']
x = np.arange(len(labels))

# Function to plot each setting
def plot_setting(datasets, stds, titles, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, (ax, data, std, title) in enumerate(zip(axes, datasets, stds, titles)):
        data_T = data.T  # Transpose: now each line is for fixed Source ε
        std_T = std.T
        for j in range(data_T.shape[0]):
            if np.any(data_T[j]):  # Skip if all zeros
                offset = (j - data_T.shape[0] // 2) * 0.05
                x_offset = x + offset
                ax.errorbar(x_offset, data_T[j], yerr=std_T[j], label=f'Source ε={labels[j]}', marker='o', capsize=3)

                if j == data_T.shape[0] - 1:
                    ax.text(
                        x_offset[-1] + 0.07, data_T[j, -1],  # Adjust Y offset as needed
                        '*', color='black', fontsize=14, ha='center'
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(10, 100)
        ax.set_xlabel("Target ε")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)

        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)

plot_setting(
    datasets=[acc, acc_few],
    stds=[std, std_few],
    titles=["DP-SHOT M→U", "DP-FewSHOT M→U"],
    filename="lineplot_m_u.png"
)

plot_setting(
    datasets=[acc2, acc2_few],
    stds=[std2, std2_few],
    titles=["DP-SHOT U→M", "DP-FewSHOT U→M"],
    filename="lineplot_u_m.png"
)

plot_setting(
    datasets=[acc3, acc3_few],
    stds=[std3, std3_few],
    titles=["DP-SHOT S→M", "DP-FewSHOT S→M"],
    filename="lineplot_s_m.png"
)
