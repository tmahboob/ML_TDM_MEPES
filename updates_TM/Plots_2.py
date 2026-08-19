import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==============================
# Load data
# ==============================

df = pd.read_csv("summary_statistics_innovations[06082026].csv")

df.columns = df.columns.str.strip()

df["Scenario"] = df["Scenario"].str.lower()
df["Detector"] = df["Detector"].str.lower()


# Select detector
detector = "isolation_forest"

df = df[df["Detector"] == detector]


# ==============================
# Radar configuration
# ==============================

windows = sorted(df["Window"].unique())

angles = np.linspace(
    0,
    2*np.pi,
    len(windows),
    endpoint=False
)

angles = np.append(
    angles,
    angles[0]
)


scenarios = [
    "random",
    "standard",
    "stealth"
]


# Different colours
colors = {
    "random": "#1f77b4",      # blue
    "standard": "#d62728",    # red
    "stealth": "#2ca02c"      # green
}


# Different line styles
styles = {
    "random": "-",
    "standard": "--",
    "stealth": ":"
}


# Different markers
markers = {
    "random": "o",
    "standard": "s",
    "stealth": "^"
}


# ==============================
# Plot radar
# ==============================

fig, ax = plt.subplots(
    figsize=(8,8),
    subplot_kw=dict(polar=True),
    dpi=300
)


for scenario in scenarios:

    data = df[
        df["Scenario"] == scenario
    ]


    fpr_values = []


    for w in windows:

        value = data[
            data["Window"] == w
        ]["FPR_Mean"].values[0]

        fpr_values.append(value)


    fpr_values = np.append(
        fpr_values,
        fpr_values[0]
    )


    ax.plot(
        angles,
        fpr_values,
        color=colors[scenario],
        linestyle=styles[scenario],
        marker=markers[scenario],
        linewidth=2.5,
        markersize=8,
        label=scenario.capitalize()
    )


    ax.fill(
        angles,
        fpr_values,
        color=colors[scenario],
        alpha=0.12
    )


# ==============================
# Formatting
# ==============================

ax.set_xticks(
    angles[:-1]
)

ax.set_xticklabels(
    [f"W={w}" for w in windows],
    fontsize=12
)


ax.set_ylim(
    0,
    0.55
)


ax.set_title(
    "False Positive Rate Comparison\n(Isolation Forest)",
    fontsize=15,
    pad=25
)


ax.grid(
    linestyle="--",
    alpha=0.4
)


ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.25,1.1),
    fontsize=12
)


plt.tight_layout()


plt.savefig(
    "FPR_radar_colour.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()