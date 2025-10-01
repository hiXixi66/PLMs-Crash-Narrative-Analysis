import os
import pandas as pd
import matplotlib.pyplot as plt

# Input folder and output folder for plots
folder = "reports/mancolll-test/class-9"
out_folder = os.path.join(folder, "plots")
os.makedirs(out_folder, exist_ok=True)  # Create folder to save images

# Fixed order of collision_type categories
collision_types = [0, 1, 2, 4, 5, 6, 9]

for fname in os.listdir(folder):
    if not fname.endswith(".xlsx"):
        continue

    fpath = os.path.join(folder, fname)
    df = pd.read_excel(fpath)

    total = len(df)
    subset = df[df["MANCOLL"] == 9]
    subset_count = len(subset)
    proportion = subset_count / total if total > 0 else 0

    # === Print the proportion of MANCOLL=9 (text output) ===
    print(f"\nFile: {fname}")
    print(f"MANCOLL=9 count: {subset_count}/{total} ({proportion*100:.2f}%)")

    # === Plot bar chart of collision_type distribution for MANCOLL=9 ===
    collision_dist = subset["collision_type"].value_counts(normalize=True)

    # Reindex to ensure fixed order and fill missing categories with 0
    collision_dist = collision_dist.reindex(collision_types, fill_value=0)

    fig, ax = plt.subplots(figsize=(5, 4))  # More compact figure
    collision_dist.plot(kind="bar", ax=ax, color="skyblue", edgecolor="black", width=0.7)

    # Add percentage labels above each bar
    for p in ax.patches:
        value = p.get_height()
        if value > 0:  # Skip 0-value bars
            ax.text(
                p.get_x() + p.get_width() / 2,  # X: center of the bar
                value + 0.01,                   # Y: slightly above the bar
                f"{value * 100:.1f}%",          # Show as percentage
                ha="center", va="bottom",
                fontsize=15,
                color="black"
            )

    # If collision_type contains 9, could add special annotation (commented out)
    # if 9 in collision_dist.index and collision_dist.loc[9] > 0:
    #     val9 = collision_dist.loc[9]

    title_name = fname[:-5] if fname.lower().endswith(".xlsx") else fname
    ymax = collision_dist.max() + 0.05  # Add margin above max value
    ax.set_ylim(0, ymax)
    ax.set_title(title_name, fontsize=14)
    ax.set_ylabel("Proportion", fontsize=14)
    ax.set_xlabel("Manner of collision", fontsize=14)
    ax.set_xticks(range(len(collision_types)))
    ax.set_xticklabels(collision_types, fontsize=14, rotation=0)

    # Save the plot (filename matches the Excel file)
    out_path = os.path.join(out_folder, fname.replace(".xlsx", ".png"))
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

print(f"\nAll bar charts have been saved to: {out_folder}")


# ==================== Pie chart part ====================

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Specify the single file to visualize
pie_file = r"reports/mancolll-test/self-con/MANCOLL-classification_LlaMA3 8B-1251-2.xlsx"

# MANCOLL categories and their meanings
mancoll_order = [0, 1, 2, 4, 5, 6, 9]
mancoll_meaning = {
    0: "Not Collision with Vehicle in Transport",
    1: "Rear-End",
    2: "Head-On",
    4: "Angle",
    5: "Sideswipe, Same Direction",
    6: "Sideswipe, Opposite Direction",
    9: "Unknown",
}

# Read Excel file
df_pie = pd.read_excel(pie_file)

# Calculate proportions by MANCOLL, fill missing classes with 0
prop = (df_pie["MANCOLL"]
        .value_counts(normalize=True)
        .reindex(mancoll_order, fill_value=0.0))

# Print proportions to console
print("\nMANCOLL proportions in file:")
for k in mancoll_order:
    print(f"{k} ({mancoll_meaning[k]}): {prop[k]*100:.2f}%")

# Pie chart setup
colors = [
    "#1f77b4",  # deep blue
    "#4c92c3",  # blue
    "#74add1",  # light blue
    "#a6cbe3",  # pale blue
    "#5aa9a6",  # teal
    "#82c0cc",  # light teal
    "#9aa7bd",  # gray-blue
]

fig, ax = plt.subplots(figsize=(7.5, 5))
wedges, _ = ax.pie(
    prop.values, startangle=20, colors=colors,
    wedgeprops=dict(linewidth=1, edgecolor="white")
)

# Threshold: only label segments >=1%
THRESH = 0.01
R_TEXT = 1.0    # Radius for label text
R_TAIL = 0.8    # Radius for leader line start
MIN_GAP = 0.06  # Minimum vertical spacing between labels

# Calculate candidate label positions
labels_xy = []
for i, w in enumerate(wedges):
    p = float(prop.values[i])
    if p < THRESH:
        continue
    ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
    x_txt, y_txt = R_TEXT * np.cos(ang), R_TEXT * np.sin(ang)
    x_tail, y_tail = R_TAIL * np.cos(ang), R_TAIL * np.sin(ang)
    ha = "left" if x_txt >= 0 else "right"
    labels_xy.append([i, p, x_txt, y_txt, x_tail, y_tail, ha])

# Adjust label y-positions to avoid overlap
labels_xy.sort(key=lambda t: t[3])  # Sort by y_txt
for j in range(1, len(labels_xy)):
    if labels_xy[j][3] - labels_xy[j - 1][3] < MIN_GAP:
        labels_xy[j][3] = labels_xy[j - 1][3] + MIN_GAP

# Draw labels with leader lines
for i, w in enumerate(wedges):
    p = float(prop.values[i])
    if p <= 0:
        continue
    ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
    x_txt, y_txt = R_TEXT * np.cos(ang), R_TEXT * np.sin(ang)
    x_tail, y_tail = R_TAIL * np.cos(ang), R_TAIL * np.sin(ang)
    ha = "left" if x_txt >= 0 else "right"

    ax.annotate(
        f"{p * 100:.1f}%", xy=(x_tail, y_tail), xytext=(x_txt, y_txt),
        ha=ha, va="center", fontsize=14, fontweight="bold", color="#2b2b2b",
        arrowprops=dict(arrowstyle="-", lw=1.2, color="#555",
                        connectionstyle="arc3,rad=0.25")
    )

ax.set_title("MANCOLL distribution", fontsize=14)
ax.axis("equal")

# Build legend
title_name = os.path.splitext(os.path.basename(pie_file))[0]
legend_labels = [
    f"{k}: {mancoll_meaning[k]}  —  {prop[k]*100:.1f}%"
    for k in mancoll_order
]
ax.legend(wedges, legend_labels, title="MANCOLL (ID: Meaning — Share)",
          loc="center left", bbox_to_anchor=(0.9, 0.5), frameon=True,
          fontsize=14, title_fontsize=15)
ax.axis("equal")

# Save figure
out_dir = os.path.join(os.path.dirname(pie_file), "plots")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, title_name.replace(" ", "_") + "_pie.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
