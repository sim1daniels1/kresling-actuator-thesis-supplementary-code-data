import pandas as pd
import matplotlib.pyplot as plt


ALIGNED_FILE = "10x_aligned_full_test.csv"
USE_SMOOTH = True
TIME_UNIT = "s"
AXIS_TITLE_FONT_SIZE = 16
TICK_LABEL_FONT_SIZE = 13
LEGEND_FONT_SIZE = 13


def load_aligned_data(path):
    df = pd.read_csv(path)

    needed_cols = ["time_s", "P_g", "P_g_smooth", "compression_pct", "compression_pct_smooth"]
    for col in needed_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=needed_cols).reset_index(drop=True)


df = load_aligned_data(ALIGNED_FILE)

if TIME_UNIT == "min":
    time_plot = df["time_s"] / 60.0
    time_label = "Time [min]"
else:
    time_plot = df["time_s"]
    time_label = "Time [s]"

compression_col = "compression_pct_smooth" if USE_SMOOTH else "compression_pct"
pressure_col = "P_g_smooth" if USE_SMOOTH else "P_g"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

ax1.plot(time_plot, df[pressure_col], linewidth=2.0, label="Pressure")
ax1.set_ylabel("Pressure [kPa]", fontsize=AXIS_TITLE_FONT_SIZE)
ax1.grid(True)
ax1.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
ax1.legend(loc="best", fontsize=LEGEND_FONT_SIZE)

ax2.plot(time_plot, df[compression_col], linewidth=2.0, color="tab:red", label="Compression")
ax2.set_xlabel(time_label, fontsize=AXIS_TITLE_FONT_SIZE)
ax2.set_ylabel("Compression [%]", fontsize=AXIS_TITLE_FONT_SIZE)
ax2.grid(True)
ax2.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
ax2.legend(loc="best", fontsize=LEGEND_FONT_SIZE)

plt.tight_layout()
plt.show()
