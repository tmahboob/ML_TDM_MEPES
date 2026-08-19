import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==============================
# Load data
# ==============================

df = pd.read_csv("summary_statistics_innovations[06082026].csv")

df.columns = df.columns.str.strip()

df["Scenario"] = df["Scenario"].str.lower()
df["Detector"] = df["Detector"].str.lower()


# ==============================
# Plot settings
# ==============================

plt.rcParams.update({
    "font.size": 12,
    "font.family": "Times New Roman"
})


colors = {
    "isolation_forest": "#222222",
    "lof": "#777777",
    "ocsvm": "#BBBBBB"
}


labels = {
    "isolation_forest": "Isolation Forest",
    "lof": "LOF",
    "ocsvm": "OCSVM"
}


scenarios = [
    "random",
    "standard",
    "stealth"
]


# ==============================
# Create one figure
# ==============================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5),
    dpi=300,
    sharey=True
)


for ax, scenario in zip(axes, scenarios):

    data = df[
        df["Scenario"] == scenario
    ]


    detectors = data["Detector"].unique()

    windows = sorted(
        data["Window"].unique()
    )


    x = np.arange(len(windows))

    width = 0.25


    for i, detector in enumerate(detectors):

        subset = data[
            data["Detector"] == detector
        ].sort_values("Window")


        means = subset["PRECISION_Mean"].values
        ci = subset["PRECISION_CI"]


        offset = (
            i - (len(detectors)-1)/2
        ) * width


        ax.bar(
            x + offset,
            means,
            width,
            yerr=ci,
            capsize=4,
            color=colors[detector],
            edgecolor="black",
            linewidth=0.8,
            error_kw={
                "elinewidth":1
            },
            label=labels[detector]
        )


    ax.set_title(
        scenario.capitalize() + " Attack",
        fontsize=14
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        windows
    )

    ax.set_xlabel(
        "Window Size"
    )


    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )


# Common y-axis label

axes[0].set_ylabel(
    "PRECISION"
)

axes[0].set_ylim(
    0,
    1.05
)


# One common legend

handles, legend_labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    legend_labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.08)
)


plt.tight_layout()


plt.savefig(
    "all_attack_scenarios_PRECISION.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# DATA
# ============================================================
# ============================================================
# IEEE STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

attack_strength = np.array([
    0.03,
    0.06,
    0.09,
    0.12,
    0.15,
    0.18,
    0.21
])

accuracy = np.array([
    0.85,
    0.886666667,
    0.93,
    0.943333333,
    0.95,
    0.953333333,
    0.95
])

precision = np.array([
    1.00,
    1.00,
    1.00,
    1.00,
    1.00,
    1.00,
    0.941176471
])

recall = np.array([
    0.25,
    0.433333333,
    0.65,
    0.716666667,
    0.75,
    0.766666667,
    0.80
])

f1 = np.array([
    0.40,
    0.604651163,
    0.787878788,
    0.834951456,
    0.857142857,
    0.867924528,
    0.864864865
])


# ============================================================
# METRICS
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]


# Each row corresponds to one attack strength
data = np.column_stack([
    accuracy,
    precision,
    recall,
    f1
])


# ============================================================
# RADAR ANGLES
# ============================================================

N = len(metrics)

angles = np.linspace(
    0,
    2 * np.pi,
    N,
    endpoint=False
)

# Close the radar polygon
angles = np.concatenate([
    angles,
    [angles[0]]
])


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(11, 11)
)

ax = plt.subplot(
    111,
    polar=True
)


# Put first metric at top
ax.set_theta_offset(
    np.pi / 2
)

# Clockwise direction
ax.set_theta_direction(
    -1
)


# ============================================================
# AXES
# ============================================================

ax.set_xticks(
    angles[:-1]
)

ax.set_xticklabels(
    metrics,
    fontsize=13,
    fontweight="bold"
)


# Fixed 0-1 scale
ax.set_ylim(
    0,
    1.0
)

ax.set_yticks([
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
])

ax.set_yticklabels([
    "0.2",
    "0.4",
    "0.6",
    "0.8",
    "1.0"
], fontsize=10)


ax.grid(
    True,
    linewidth=0.8,
    alpha=0.6
)


# ============================================================
# PLOT EACH ATTACK STRENGTH
# ============================================================

for i, strength in enumerate(attack_strength):

    values = data[i]

    values = np.concatenate([
        values,
        [values[0]]
    ])


    ax.plot(
        angles,
        values,
        linewidth=2,
        marker="o",
        markersize=5,
        label=f"Attack strength = {strength:.2f}"
    )


# ============================================================
# TITLE
# ============================================================

#plt.title(
  #  "OCSVM Performance Across Attack Strengths",
   # fontsize=17,
    #fontweight="bold",
    #pad=30
#)


# ============================================================
# LEGEND
# ============================================================


legend = plt.legend(
    title="Attack Strength",
    loc="lower center",
    bbox_to_anchor=(0.5, -0.24),
    ncol=2,                  # Better readability than 4 columns
    fontsize=12,
    title_fontsize=12,
    frameon=False,
    handlelength=2.0,
    columnspacing=1.2,
    handletextpad=0.6,
    borderaxespad=0.4
)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout()

plt.savefig(
    "ocsvm_attack_strength_spider.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "ocsvm_attack_strength_spider.pdf",
    bbox_inches="tight"
)


# ============================================================
# DISPLAY
# ============================================================

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.font_manager as fm

# ============================================================
# IEEE-9 BUS ATTACK: RECALL AND F1
# Grayscale bar chart - Times New Roman
# ============================================================

# ------------------------------------------------------------
# 1. DATA
# ------------------------------------------------------------
data = {
    "Bus": [1, 2, 3, 4, 5, 6, 7, 8, 9],

    "Recall": [
        0.466667,
        0.400000,
        0.633333,
        0.683333,
        0.233333,
        0.383333,
        0.233333,
        0.583333,
        0.616667
    ],

    "F1": [
        0.636364,
        0.571429,
        0.775510,
        0.811881,
        0.378378,
        0.554217,
        0.378378,
        0.736842,
        0.762887
    ]
}

df = pd.DataFrame(data)

# ------------------------------------------------------------
# 2. OUTPUT DIRECTORY
# ------------------------------------------------------------
output_dir = Path(
    r"C:\Users\tm120a\OneDrive - University of Glasgow\Desktop\NetLab2024"
    r"\Implementation\CARAz_work\DAME-FDIA-main\DAME-FDIA-main"
    r"\bus_attack_results\plots"
)

output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 3. FIND TIMES NEW ROMAN
# ------------------------------------------------------------
times_fonts = [
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\timesbd.ttf",
    r"C:\Windows\Fonts\timesi.ttf",
]

times_font = None

for font_path in times_fonts:
    if Path(font_path).exists():
        times_font = font_path
        break

if times_font:
    font_prop = fm.FontProperties(fname=times_font)
    font_name = font_prop.get_name()
    plt.rcParams["font.family"] = font_name
    print("Using font:", font_name)
else:
    print("WARNING: Times New Roman not found.")
    print("Using Liberation Serif as fallback.")
    plt.rcParams["font.family"] = "Liberation Serif"

# ------------------------------------------------------------
# 4. PLOT
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(df["Bus"]))
width = 0.36

# Grayscale only
recall_bars = ax.bar(
    x - width / 2,
    df["Recall"],
    width,
    label="Recall",
    color="0.65",
    edgecolor="black",
    linewidth=1.0
)

f1_bars = ax.bar(
    x + width / 2,
    df["F1"],
    width,
    label="F1",
    color="0.30",
    edgecolor="black",
    linewidth=1.0,
    hatch="//"
)

# ------------------------------------------------------------
# 5. AXES
# ------------------------------------------------------------
ax.set_xlabel(
    "Attack Bus",
    fontsize=15,
    fontweight="bold"
)

ax.set_ylabel(
    "Score",
    fontsize=15,
    fontweight="bold"
)

ax.set_title(
    "Recall and F1 Score for IEEE-9 Bus Attack Scenarios",
    fontsize=16,
    fontweight="bold"
)

ax.set_xticks(x)
ax.set_xticklabels(
    df["Bus"],
    fontsize=13
)

ax.set_ylim(0, 1.0)

ax.set_yticks(np.arange(0, 1.01, 0.1))

ax.tick_params(
    axis="both",
    labelsize=12
)

# Horizontal grid
ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.5
)

ax.set_axisbelow(True)

# ------------------------------------------------------------
# 6. LEGEND
# ------------------------------------------------------------
ax.legend(
    fontsize=12,
    loc="upper right",
    frameon=True,
    edgecolor="black"
)

# ------------------------------------------------------------
# 7. VALUE LABELS
# ------------------------------------------------------------
for bars in [recall_bars, f1_bars]:

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.018,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

# ------------------------------------------------------------
# 8. AXIS BORDER
# ------------------------------------------------------------
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

# ------------------------------------------------------------
# 9. LAYOUT
# ------------------------------------------------------------
plt.tight_layout()

# ------------------------------------------------------------
# 10. SAVE
# ------------------------------------------------------------
png_file = output_dir / "ieee9_recall_f1_by_attack_bus.png"
pdf_file = output_dir / "ieee9_recall_f1_by_attack_bus.pdf"

plt.savefig(
    png_file,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    pdf_file,
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# 11. PRINT RESULTS
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("PLOT SAVED")
print("=" * 70)

print("PNG:")
print(png_file)

print("\nPDF:")
print(pdf_file)

print("\nData:")
print(df.to_string(index=False))

print("=" * 70)