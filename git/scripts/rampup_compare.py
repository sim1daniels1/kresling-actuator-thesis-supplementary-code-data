import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# SETTINGS
# =========================================================
INITIAL_FILE = "aligned_initial_ramp.csv"
FINAL_FILE = "aligned_final_ramp.csv"

# Reference height for 0% compression:
# taken from the initial low-pressure part of the INITIAL ramp
ZERO_KPA_MAX = 2.0          # define "near zero pressure" for reference region
REF_FALLBACK_N = 20         # use first N rows if too few points satisfy zero-pressure condition

# Optional smoothing for plotting
SMOOTH_WINDOW = 5

# Relative time options
USE_RELATIVE_TIME = True    # start each ramp at t = 0
TIME_UNIT = "s"             # "s" or "min"
# =========================================================


def rolling_mean(series, window):
    return series.rolling(window=window, center=True, min_periods=1).mean()


def compute_reference_height(df_init):
    # Prefer rows near 0 kPa in the initial ramp
    zero_mask = df_init["P_g_smooth"] <= ZERO_KPA_MAX

    if zero_mask.sum() >= 3:
        h_ref = df_init.loc[zero_mask, "Height_yellow_px_smooth"].mean()
    else:
        h_ref = df_init["Height_yellow_px_smooth"].iloc[:REF_FALLBACK_N].mean()

    return h_ref


def prepare_ramp(df, h_ref):
    df = df.copy()

    # Compression normalized to initial specimen length from first ramp
    df["compression_pct"] = (h_ref - df["Height_yellow_px"]) / h_ref * 100.0
    df["compression_pct_smooth"] = rolling_mean(df["compression_pct"], SMOOTH_WINDOW)

    # Relative time if desired
    if USE_RELATIVE_TIME:
        df["time_plot"] = df["time_s"] - df["time_s"].iloc[0]
    else:
        df["time_plot"] = df["time_s"]

    if TIME_UNIT == "min":
        df["time_plot"] = df["time_plot"] / 60.0
        time_label = "Time [min]"
    else:
        time_label = "Time [s]"

    return df, time_label


# =========================================================
# LOAD DATA
# =========================================================
init_df = pd.read_csv(INITIAL_FILE)
final_df = pd.read_csv(FINAL_FILE)

# Make sure needed columns are numeric
needed_cols = [
    "time_s", "P_g", "P_g_smooth", "rampTarget_g",
    "Height_yellow_px", "Height_yellow_px_smooth"
]

for col in needed_cols:
    init_df[col] = pd.to_numeric(init_df[col], errors="coerce")
    final_df[col] = pd.to_numeric(final_df[col], errors="coerce")

init_df = init_df.dropna(subset=needed_cols).reset_index(drop=True)
final_df = final_df.dropna(subset=needed_cols).reset_index(drop=True)

# =========================================================
# REFERENCE HEIGHT FROM INITIAL RAMP ONLY
# =========================================================
h_ref = compute_reference_height(init_df)
print(f"Reference height from initial ramp at ~0 kPa: {h_ref:.3f} px")

# =========================================================
# PREPARE DATA
# =========================================================
init_df, time_label = prepare_ramp(init_df, h_ref)
final_df, _ = prepare_ramp(final_df, h_ref)

print(f"Initial ramp max compression: {init_df['compression_pct_smooth'].max():.2f}%")
print(f"Final ramp max compression:   {final_df['compression_pct_smooth'].max():.2f}%")

# =========================================================
# PLOT OVERLAY
# =========================================================
fig, ax1 = plt.subplots(figsize=(11, 5))

# Compression on left axis
ax1.plot(
    init_df["time_plot"],
    init_df["compression_pct_smooth"],
    linewidth=2,
    label="Initial ramp compression"
)
ax1.plot(
    final_df["time_plot"],
    final_df["compression_pct_smooth"],
    linewidth=2,
    label="Final ramp compression"
)
ax1.set_xlabel(time_label)
ax1.set_ylabel("Axial compression [%]")
ax1.grid(True)

# Pressure on right axis
ax2 = ax1.twinx()
ax2.plot(
    init_df["time_plot"],
    init_df["P_g_smooth"],
    "--",
    linewidth=2,
    label="Initial ramp pressure"
)
ax2.plot(
    final_df["time_plot"],
    final_df["P_g_smooth"],
    "--",
    linewidth=2,
    label="Final ramp pressure"
)
ax2.plot(
    init_df["time_plot"],
    init_df["rampTarget_g"],
    ":",
    linewidth=1.5,
    label="Initial ramp target"
)
ax2.plot(
    final_df["time_plot"],
    final_df["rampTarget_g"],
    ":",
    linewidth=1.5,
    label="Final ramp target"
)
ax2.set_ylabel("Gauge pressure [kPa]")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.title("Initial vs final ramp overlay")
plt.tight_layout()
plt.show()
