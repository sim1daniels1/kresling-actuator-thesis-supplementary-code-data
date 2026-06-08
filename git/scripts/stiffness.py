import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle

# ============================================================
# USER INPUTS - fill these values for your specimens
# ============================================================


GEOM = {
    "10x": {
        "A":  np.pi * (1400e-6)**2 ,   # m^2
        "L0": 5737.5e-6,
        "h0": 5737.5e-6,
        "P0_kPa": 100
    },
    "20x": {
        "A":  np.pi * (700e-6)**2 ,    # m^2
        "L0": 2868.75e-6,    # m
        "h0": 2868.75e-6,    # m (usually same as L0)
        "P0_kPa": 100 # baseline pressure
    },
    "40x": {
        "A":  np.pi * (350e-6)**2 ,   # m^2
        "L0": 1434.375e-6,
        "h0": 1434.375e-6,
        "P0_kPa": 100
}
}


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "all_error_plot_points.csv")

INTERVAL_TABLE_CSV = os.path.join(BASE_DIR, "nonlinear_stiffness_intervals.csv")
GLOBAL_TABLE_CSV = os.path.join(BASE_DIR, "global_stiffness_summary.csv")
STIFFNESS_PLOT = os.path.join(BASE_DIR, "nonlinear_stiffness_grouped_bars.png")
EQUIVALENT_STIFFNESS_STACKED_PLOT = os.path.join(BASE_DIR, "equivalent_stiffness_stacked_bars.png")
AIR_FRACTION_PLOT = os.path.join(BASE_DIR, "trapped_air_fraction_interval_stiffness.png")
RELATIVE_STIFFNESS_STACKED_PLOT = os.path.join(BASE_DIR, "relative_stiffness_contributions_stacked_bars.png")
AIR_PRESSURE_PLOT = os.path.join(BASE_DIR, "trapped_air_pressure_vs_compression.png")
AIR_STIFFNESS_PLOT = os.path.join(BASE_DIR, "trapped_air_stiffness_vs_compression.png")
STIFFNESS_COMPONENTS_PLOT = os.path.join(BASE_DIR, "stiffness_components_by_specimen.png")

# Pressure values in all_error_plot_points.csv are gauge pressures.
# P0_kPa in GEOM is the absolute trapped-air pressure at 0 kPa gauge
# pressure, so the air correction uses P_abs = P0_kPa + P_gauge.
#
# GAS_GAMMA = 1.0 gives an isothermal trapped-air correction.
# Use 1.4 if you want an adiabatic correction instead.
GAS_GAMMA = 1.0
INCLUDE_ORIGIN_FOR_INTERVALS = True

AXIS_LABEL_FONTSIZE = 17
TICK_LABEL_FONTSIZE = 13
TITLE_FONTSIZE = 17
LEGEND_FONTSIZE = 15


def style_plot_axis(ax):
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)


class HandlerSplitStiffnessBar(HandlerBase):
    """Legend swatch with solid actuator and transparent air segments."""

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        color = orig_handle.get_facecolor()
        lower = Rectangle(
            (xdescent, ydescent),
            width,
            height * 0.58,
            facecolor=color,
            edgecolor="0.15",
            linewidth=0.8,
            alpha=0.95,
            transform=trans,
        )
        upper = Rectangle(
            (xdescent, ydescent + height * 0.58),
            width,
            height * 0.42,
            facecolor=color,
            edgecolor="0.15",
            linewidth=0.8,
            alpha=0.35,
            transform=trans,
        )
        return [lower, upper]


# ============================================================
# DATA LOADING
# ============================================================

def load_pressure_compression_points(csv_path):
    df = pd.read_csv(csv_path)

    rename_map = {
        "test_label": "specimen",
        "mean_pressure_kpa": "pressure_kPa",
        "mean_deformation_pct": "compression_pct",
    }
    df = df.rename(columns={key: value for key, value in rename_map.items() if key in df.columns})

    if "target_kpa" in df.columns:
        df["target_kpa"] = pd.to_numeric(df["target_kpa"], errors="coerce")
    else:
        df["target_kpa"] = pd.to_numeric(df["pressure_kPa"], errors="coerce").round(-1)

    required = {"specimen", "target_kpa", "pressure_kPa", "compression_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df = df.dropna(subset=["specimen", "target_kpa", "pressure_kPa", "compression_pct"]).copy()
    df["specimen"] = df["specimen"].astype(str)
    df["target_kpa"] = pd.to_numeric(df["target_kpa"], errors="coerce")
    df["pressure_kPa"] = pd.to_numeric(df["pressure_kPa"], errors="coerce")
    df["compression_pct"] = pd.to_numeric(df["compression_pct"], errors="coerce")
    df = df.dropna(subset=["target_kpa", "pressure_kPa", "compression_pct"])

    missing_geom = sorted(set(df["specimen"]) - set(GEOM))
    if missing_geom:
        raise ValueError(f"Missing GEOM entries for specimens: {missing_geom}")

    return df.sort_values(["specimen", "target_kpa"]).reset_index(drop=True)


# ============================================================
# STIFFNESS CALCULATIONS
# ============================================================

def equivalent_stiffness_from_slope(slope_kpa_per_pct, geom):
    """Convert dP/d(compression %) to equivalent axial stiffness [N/m]."""
    slope_pa_per_strain = slope_kpa_per_pct * 1e3 * 100.0
    return geom["A"] * slope_pa_per_strain / geom["L0"]


def trapped_air_stiffness(pressure_gauge_kpa, compression_pct, geom):
    """Pressure-dependent tangent stiffness of trapped air [N/m]."""
    strain = compression_pct / 100.0
    current_height = geom["h0"] * max(1.0 - strain, 1e-6)
    # The exported pressure is gauge pressure; add the 0-gauge absolute baseline.
    pressure_abs_pa = (geom["P0_kPa"] + pressure_gauge_kpa) * 1e3
    return GAS_GAMMA * pressure_abs_pa * geom["A"] / current_height


def stiffness_from_slope(slope_kpa_per_pct, pressure_gauge_kpa, compression_pct, geom):
    k_eq = equivalent_stiffness_from_slope(slope_kpa_per_pct, geom)
    k_air = trapped_air_stiffness(pressure_gauge_kpa, compression_pct, geom)
    k_kresling = k_eq - k_air
    return k_eq, k_air, k_kresling


def linear_fit_summary(specimen, data):
    geom = GEOM[specimen]
    c = data["compression_pct"].to_numpy(dtype=float)
    p = data["pressure_kPa"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(c, p, 1)
    pressure_mid = float(np.mean(p))
    compression_mid = float(np.mean(c))
    k_eq, k_air, k_kresling = stiffness_from_slope(slope, pressure_mid, compression_mid, geom)

    return {
        "specimen": specimen,
        "fit_type": "global_linear",
        "slope_kpa_per_pct": slope,
        "intercept_kpa": intercept,
        "pressure_for_air_kpa": pressure_mid,
        "compression_for_air_pct": compression_mid,
        "k_equivalent_N_per_m": k_eq,
        "k_air_N_per_m": k_air,
        "k_kresling_corrected_N_per_m": k_kresling,
    }


def interval_stiffness_rows(specimen, data):
    geom = GEOM[specimen]
    points = data[["target_kpa", "pressure_kPa", "compression_pct"]].copy()

    if INCLUDE_ORIGIN_FOR_INTERVALS:
        origin = pd.DataFrame([{
            "target_kpa": 0.0,
            "pressure_kPa": 0.0,
            "compression_pct": 0.0,
        }])
        points = pd.concat([origin, points], ignore_index=True)

    rows = []
    for i in range(1, len(points)):
        target0 = float(points.loc[i - 1, "target_kpa"])
        target1 = float(points.loc[i, "target_kpa"])
        p0 = float(points.loc[i - 1, "pressure_kPa"])
        p1 = float(points.loc[i, "pressure_kPa"])
        c0 = float(points.loc[i - 1, "compression_pct"])
        c1 = float(points.loc[i, "compression_pct"])

        dp = p1 - p0
        dc = c1 - c0
        if np.isclose(dc, 0.0):
            continue

        slope = dp / dc
        pressure_mid = 0.5 * (p0 + p1)
        compression_mid = 0.5 * (c0 + c1)
        k_eq, k_air, k_kresling = stiffness_from_slope(slope, pressure_mid, compression_mid, geom)

        rows.append({
            "specimen": specimen,
            "interval": f"{target0:.0f}-{target1:.0f} kPa",
            "target_start_kpa": target0,
            "target_end_kpa": target1,
            "target_mid_kpa": 0.5 * (target0 + target1),
            "pressure_start_kpa": p0,
            "pressure_end_kpa": p1,
            "pressure_mid_kpa": pressure_mid,
            "compression_start_pct": c0,
            "compression_end_pct": c1,
            "compression_mid_pct": compression_mid,
            "slope_kpa_per_pct": slope,
            "k_equivalent_N_per_m": k_eq,
            "k_air_N_per_m": k_air,
            "k_kresling_corrected_N_per_m": k_kresling,
        })

    return rows


# ============================================================
# OUTPUT
# ============================================================

df = load_pressure_compression_points(INPUT_CSV)

global_rows = []
interval_rows = []

for specimen, data in df.groupby("specimen", sort=False):
    data = data.sort_values("target_kpa").reset_index(drop=True)
    global_rows.append(linear_fit_summary(specimen, data))
    interval_rows.extend(interval_stiffness_rows(specimen, data))

global_df = pd.DataFrame(global_rows)
interval_df = pd.DataFrame(interval_rows)

global_df.to_csv(GLOBAL_TABLE_CSV, index=False)
interval_df.to_csv(INTERVAL_TABLE_CSV, index=False)

print("\nGlobal linear stiffness with trapped-air correction")
print(f"Air model: P_abs = P0_kPa + P_gauge, gamma = {GAS_GAMMA:g}")
print(global_df[[
    "specimen",
    "slope_kpa_per_pct",
    "k_equivalent_N_per_m",
    "k_air_N_per_m",
    "k_kresling_corrected_N_per_m",
]].to_string(index=False, float_format=lambda value: f"{value:.4g}"))

print("\nInterval nonlinear stiffness with trapped-air correction")
print(interval_df[[
    "specimen",
    "interval",
    "slope_kpa_per_pct",
    "k_equivalent_N_per_m",
    "k_air_N_per_m",
    "k_kresling_corrected_N_per_m",
]].to_string(index=False, float_format=lambda value: f"{value:.4g}"))

print("\nLaTeX rows for global corrected stiffness:")
print("Specimen & $m$ [kPa/\\%] & $k_{eq}$ [N/m] & $k_{air}$ [N/m] & $k_K$ [N/m] \\\\")
for _, row in global_df.iterrows():
    print(
        f"{row['specimen']} & {row['slope_kpa_per_pct']:.3f} & "
        f"{row['k_equivalent_N_per_m']:.2e} & {row['k_air_N_per_m']:.2e} & "
        f"{row['k_kresling_corrected_N_per_m']:.2e} \\\\"
    )

print("\nLaTeX interval stiffness table:")
interval_table = interval_df.pivot(
    index="specimen",
    columns="interval",
    values="k_kresling_corrected_N_per_m",
)
interval_order = interval_df[["interval", "target_mid_kpa"]].drop_duplicates()
interval_order = interval_order.sort_values("target_mid_kpa")["interval"].tolist()
interval_table = interval_table.reindex(columns=interval_order)

missing_interval_values = interval_table.isna()
if missing_interval_values.any().any():
    missing_pairs = [
        f"{specimen} / {interval}"
        for specimen, row in missing_interval_values.iterrows()
        for interval, is_missing in row.items()
        if is_missing
    ]
    print("\nWarning: missing interval stiffness values:")
    print(", ".join(missing_pairs))

print(
    "Specimen & "
    + " & ".join(interval_table.columns)
    + " \\\\"
)
for specimen, row in interval_table.iterrows():
    values = " & ".join(f"{value:.0f}" for value in row.to_numpy(dtype=float))
    print(f"{specimen} & {values} \\\\")

air_fraction_df = interval_df.copy()
air_fraction_df["k_air_fraction_pct"] = (
    100.0
    * air_fraction_df["k_air_N_per_m"]
    / air_fraction_df["k_equivalent_N_per_m"]
)
air_fraction_df["k_K_fraction_pct"] = (
    100.0
    * air_fraction_df["k_kresling_corrected_N_per_m"]
    / air_fraction_df["k_equivalent_N_per_m"]
)

print("\nTrapped-air fraction of equivalent interval stiffness")
print(air_fraction_df[[
    "specimen",
    "interval",
    "k_kresling_corrected_N_per_m",
    "k_air_N_per_m",
    "k_equivalent_N_per_m",
    "k_air_fraction_pct",
]].rename(columns={
    "k_kresling_corrected_N_per_m": "k_K_N_per_m",
    "k_air_N_per_m": "k_air_N_per_m",
    "k_equivalent_N_per_m": "k_eq_N_per_m",
    "k_air_fraction_pct": "k_air/k_eq_pct",
}).to_string(index=False, float_format=lambda value: f"{value:.2f}"))

fig, ax = plt.subplots(figsize=(9.4, 5.4))

specimens = list(interval_table.index)
intervals = list(interval_table.columns)
# Remove redundant unit suffix from interval labels (e.g. "0-10 kPa" -> "0-10")
display_labels = [s.replace(" kPa", "") for s in intervals]
x = np.arange(len(intervals))
bar_width = 0.72 / max(len(specimens), 1)

colors = {
    "10x": "#1f77b4",
    "20x": "#1f9d3a",
    "40x": "#ef3b32",
}

for i, specimen in enumerate(specimens):
    offset = (i - (len(specimens) - 1) / 2.0) * bar_width
    ax.bar(
        x + offset,
        interval_table.loc[specimen].to_numpy(dtype=float),
        width=bar_width,
        label=specimen,
        color=colors.get(specimen),
        edgecolor="0.15",
        linewidth=0.8,
        alpha=0.95,
    )

ax.set_xlabel("Pressure interval [kPa]", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_ylabel("Kresling stiffness [N/m]", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_xticks(x)
ax.set_xticklabels(display_labels)
style_plot_axis(ax)
ax.set_axisbelow(True)
ax.grid(True, axis="y", alpha=0.25, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=True, edgecolor="0.2", facecolor="white", fontsize=LEGEND_FONTSIZE)
plt.tight_layout()
plt.savefig(STIFFNESS_PLOT, dpi=250)
plt.show()

fig, ax = plt.subplots(figsize=(9.4, 5.4))

air_table = interval_df.pivot(
    index="specimen",
    columns="interval",
    values="k_air_N_per_m",
).reindex(index=specimens, columns=intervals)
actuator_table = interval_df.pivot(
    index="specimen",
    columns="interval",
    values="k_kresling_corrected_N_per_m",
).reindex(index=specimens, columns=intervals)

legend_handles = []
legend_labels = []

for i, specimen in enumerate(specimens):
    offset = (i - (len(specimens) - 1) / 2.0) * bar_width
    specimen_color = colors.get(specimen)
    actuator_bars = ax.bar(
        x + offset,
        actuator_table.loc[specimen].to_numpy(dtype=float),
        width=bar_width,
        color=specimen_color,
        edgecolor="0.15",
        linewidth=0.8,
        alpha=0.95,
    )
    air_bars = ax.bar(
        x + offset,
        air_table.loc[specimen].to_numpy(dtype=float),
        width=bar_width,
        bottom=actuator_table.loc[specimen].to_numpy(dtype=float),
        color=specimen_color,
        edgecolor="0.15",
        linewidth=0.8,
        alpha=0.35,
    )

    legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=specimen_color))
    legend_labels.append(specimen)

ax.set_xlabel("Pressure interval [kPa]", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_ylabel("Equivalent stiffness [N/m]", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_xticks(x)
ax.set_xticklabels(display_labels)
style_plot_axis(ax)
ax.set_axisbelow(True)
ax.grid(True, axis="y", alpha=0.25, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(
    legend_handles,
    legend_labels,
    frameon=True,
    edgecolor="0.2",
    facecolor="white",
    fontsize=LEGEND_FONTSIZE,
    handler_map={Rectangle: HandlerSplitStiffnessBar()},
)
plt.tight_layout()
plt.savefig(EQUIVALENT_STIFFNESS_STACKED_PLOT, dpi=250)
plt.show()

fig, ax = plt.subplots(figsize=(7.8, 5.0))

for specimen in specimens:
    data = air_fraction_df[air_fraction_df["specimen"] == specimen].copy()
    data = data.sort_values("target_mid_kpa")

    ax.plot(
        data["target_mid_kpa"],
        data["k_air_fraction_pct"],
        marker="o",
        markersize=6,
        linewidth=2.2,
        color=colors.get(specimen),
        label=specimen,
    )

ax.axhline(
    50.0,
    color="0.25",
    linewidth=1.4,
    linestyle="--",
    alpha=0.8,
    label="50%",
)
ax.set_xlabel("Pressure interval midpoint [kPa]", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_ylabel("Trapped-air contribution to equivalent stiffness [%]", fontsize=AXIS_LABEL_FONTSIZE)
style_plot_axis(ax)
ax.set_axisbelow(True)
ax.grid(True, alpha=0.25, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=True, edgecolor="0.2", facecolor="white", fontsize=LEGEND_FONTSIZE)
plt.tight_layout()
plt.savefig(AIR_FRACTION_PLOT, dpi=300, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, len(specimens), figsize=(13.5, 4.8), sharey=True)
if len(specimens) == 1:
    axes = [axes]

component_colors = {
    "k_K": "#22cd04",
    "k_air": "#5b6ee1",
}

for ax, specimen in zip(axes, specimens):
    data = air_fraction_df[air_fraction_df["specimen"] == specimen].copy()
    data = data.sort_values("target_mid_kpa")
    x_subplot = np.arange(len(data))
    midpoint_labels = data["target_mid_kpa"].map(lambda value: f"{value:.0f}")

    k_k_fraction = data["k_K_fraction_pct"].to_numpy(dtype=float)
    k_air_fraction = data["k_air_fraction_pct"].to_numpy(dtype=float)

    ax.bar(
        x_subplot,
        k_k_fraction,
        width=0.72,
        color=component_colors["k_K"],
        edgecolor="0.15",
        linewidth=0.8,
        label=r"$k_\mathrm{K}$",
    )
    ax.bar(
        x_subplot,
        k_air_fraction,
        width=0.72,
        bottom=k_k_fraction,
        color=component_colors["k_air"],
        edgecolor="0.15",
        linewidth=0.8,
        label=r"$k_\mathrm{air}$",
    )
    ax.axhline(50.0, color="0.25", linewidth=1.2, linestyle="--", alpha=0.8)

    ax.set_title(specimen, fontweight="bold", fontsize=TITLE_FONTSIZE)
    ax.set_xticks(x_subplot)
    ax.set_xticklabels(midpoint_labels)
    ax.set_ylim(0.0, 100.0)
    style_plot_axis(ax)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Contribution to equivalent stiffness [%]", fontsize=AXIS_LABEL_FONTSIZE)
fig.supxlabel("Pressure interval midpoint [kPa]", fontsize=AXIS_LABEL_FONTSIZE, y=0.04)
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=True,
    edgecolor="0.2",
    facecolor="white",
    fontsize=LEGEND_FONTSIZE,
    bbox_to_anchor=(0.5, 1.02),
)
plt.tight_layout(rect=(0, 0.05, 1, 0.94))
plt.savefig(RELATIVE_STIFFNESS_STACKED_PLOT, dpi=300, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, len(specimens), figsize=(13.5, 4.6), sharey=True)
if len(specimens) == 1:
    axes = [axes]

component_styles = [
    ("k_equivalent_N_per_m", r"$k_\mathrm{eq}$", "#595959"),
    ("k_air_N_per_m", r"$k_\mathrm{air}$", "#d55e00"),
    ("k_kresling_corrected_N_per_m", r"$k_\mathrm{K}$", "#0072b2"),
]

for ax, specimen in zip(axes, specimens):
    data = interval_df[interval_df["specimen"] == specimen].copy()
    data = data.sort_values("target_mid_kpa")

    for column, label, color in component_styles:
        ax.plot(
            data["target_mid_kpa"],
            data[column],
            marker="o",
            linewidth=2.0,
            color=color,
            label=label,
        )

    ax.set_title(specimen, fontweight="bold", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Pressure interval midpoint [kPa]", fontsize=AXIS_LABEL_FONTSIZE)
    style_plot_axis(ax)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Interval stiffness [N/m]", fontsize=AXIS_LABEL_FONTSIZE)
axes[-1].legend(loc="upper left", frameon=True, fontsize=LEGEND_FONTSIZE)
fig.suptitle(
    "Equivalent, trapped-air, and corrected interval stiffness",
    y=1.02,
    fontsize=TITLE_FONTSIZE,
)
plt.tight_layout()
plt.savefig(STIFFNESS_COMPONENTS_PLOT, dpi=250, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(7.4, 4.8))

max_compression = float(df["compression_pct"].max())
compression_limit = min(max(60.0, np.ceil(max_compression / 5.0) * 5.0), 95.0)
compression_curve = np.linspace(0.0, compression_limit, 400)

for specimen in sorted(GEOM):
    p0_kpa = GEOM[specimen]["P0_kPa"]
    strain = compression_curve / 100.0
    p_air_abs_kpa = p0_kpa / np.power(1.0 - strain, GAS_GAMMA)
    p_air_gauge_kpa = p_air_abs_kpa - p0_kpa

    ax.plot(
        compression_curve,
        p_air_gauge_kpa,
        linewidth=2.0,
        color=colors.get(specimen),
        label=specimen,
    )

ax.set_xlabel("Compression [%]", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_ylabel("Trapped-air gauge pressure [kPa]", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_title("Idealized trapped-air pressure increase", fontsize=TITLE_FONTSIZE)
style_plot_axis(ax)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=LEGEND_FONTSIZE)
plt.tight_layout()
plt.savefig(AIR_PRESSURE_PLOT, dpi=250)
plt.show()

fig, ax = plt.subplots(figsize=(7.4, 4.8))

for specimen in sorted(GEOM):
    geom = GEOM[specimen]
    strain = compression_curve / 100.0
    height = geom["h0"] * (1.0 - strain)
    p_air_abs_pa = geom["P0_kPa"] * 1e3 / np.power(1.0 - strain, GAS_GAMMA)
    k_air = GAS_GAMMA * p_air_abs_pa * geom["A"] / height

    ax.plot(
        compression_curve,
        k_air,
        linewidth=2.0,
        color=colors.get(specimen),
        label=specimen,
    )

ax.set_xlabel("Axial compression [%]", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_ylabel("Estimated air stiffness [N/m]", fontsize=AXIS_LABEL_FONTSIZE)
# Match grouped-bar styling: subtle dashed grid, hide top/right spines, framed legend
style_plot_axis(ax)
ax.set_axisbelow(True)
ax.grid(True, alpha=0.25, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=True, edgecolor="0.2", facecolor="white", fontsize=LEGEND_FONTSIZE)
plt.tight_layout()
plt.savefig(AIR_STIFFNESS_PLOT, dpi=250)
plt.show()

print(f"\nSaved global summary: {GLOBAL_TABLE_CSV}")
print(f"Saved interval table: {INTERVAL_TABLE_CSV}")
print(f"Saved plot: {STIFFNESS_PLOT}")
print(f"Saved equivalent stiffness stacked plot: {EQUIVALENT_STIFFNESS_STACKED_PLOT}")
print(f"Saved trapped-air fraction plot: {AIR_FRACTION_PLOT}")
print(f"Saved relative stiffness contribution plot: {RELATIVE_STIFFNESS_STACKED_PLOT}")
print(f"Saved stiffness components plot: {STIFFNESS_COMPONENTS_PLOT}")
print(f"Saved trapped-air pressure plot: {AIR_PRESSURE_PLOT}")
print(f"Saved trapped-air stiffness plot: {AIR_STIFFNESS_PLOT}")
