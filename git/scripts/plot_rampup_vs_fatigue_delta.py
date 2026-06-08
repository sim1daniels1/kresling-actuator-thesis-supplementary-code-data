from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FATIGUE_DIR = PROJECT_DIR / "fatigue"

FIGURE_FILE = BASE_DIR / "rampup_vs_fatigue_delta.png"

SCALES = [
    {
        "label": "10x",
        "display": "10x",
        "color": "tab:blue",
        "fatigue_dir": FATIGUE_DIR / "10x",
        "fatigue_prefix": "10x",
    },
    {
        "label": "20x",
        "display": "20x",
        "color": "tab:green",
        "fatigue_dir": FATIGUE_DIR / "20x_new",
        "fatigue_prefix": "20x_new",
    },
    {
        "label": "40x",
        "display": "40x",
        "color": "tab:red",
        "fatigue_dir": FATIGUE_DIR / "40x_new",
        "fatigue_prefix": "40x_new",
    },
]

INITIAL_RAMP_STATE = "RAMP_PRE"
FINAL_RAMP_STATE = "RAMP_POST"
HOLD_START_TRIM_FRAC = 0.35
HOLD_START_TRIM_MIN_POINTS = 3
EDGE_TRIM_FRAC = 0.15
EDGE_TRIM_MIN_POINTS = 2
MIN_STABLE_POINTS = 5

WORKING_PRESSURES = np.arange(0, 101, 10, dtype=float)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "savefig.dpi": 600,
    "axes.grid": True,
    "grid.alpha": 0.22,
})


def rampup_full_test_file(scale_label):
    scale_dir = BASE_DIR / scale_label
    preferred = scale_dir / f"{scale_label}_aligned_full_test.csv"
    if preferred.exists():
        return preferred

    legacy = scale_dir / "aligned_full_test.csv"
    if legacy.exists():
        return legacy

    raise FileNotFoundError(f"No aligned full ramp-up CSV found for {scale_label}.")


def load_rampup_data(scale_label):
    path = rampup_full_test_file(scale_label)
    df = pd.read_csv(path)

    required = [
        "time_s",
        "state",
        "P_g",
        "P_g_smooth",
        "rampTarget_g",
        "rampHold",
        "compression_pct",
        "compression_pct_smooth",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    for col in required:
        if col != "state":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["state"] = df["state"].astype(str)

    return df


def trim_stable_region(seg):
    if seg is None or seg.empty:
        return seg

    n_total = len(seg)
    trim_n = max(EDGE_TRIM_MIN_POINTS, int(round(n_total * EDGE_TRIM_FRAC)))
    max_trim = max(0, (n_total - MIN_STABLE_POINTS) // 2)
    trim_n = min(trim_n, max_trim)
    stable = seg.iloc[trim_n:n_total - trim_n].copy() if trim_n > 0 else seg.copy()
    if len(stable) < MIN_STABLE_POINTS:
        stable = seg.copy()

    n_stable = len(stable)
    start_trim_n = max(HOLD_START_TRIM_MIN_POINTS, int(round(n_stable * HOLD_START_TRIM_FRAC)))
    max_start_trim = max(0, n_stable - MIN_STABLE_POINTS)
    start_trim_n = min(start_trim_n, max_start_trim)
    if start_trim_n > 0:
        stable = stable.iloc[start_trim_n:].copy()

    return stable.reset_index(drop=True)


def hold_segments(df, state):
    hold_df = df[(df["state"] == state) & (df["rampHold"] == 1)].copy()
    if hold_df.empty:
        return []

    hold_df["orig_index"] = hold_df.index
    new_segment = (
        hold_df["rampTarget_g"].ne(hold_df["rampTarget_g"].shift())
        | hold_df["orig_index"].diff().fillna(1).ne(1)
    )
    hold_df["segment_id"] = new_segment.cumsum()
    return [seg.drop(columns=["orig_index", "segment_id"]).reset_index(drop=True) for _, seg in hold_df.groupby("segment_id", sort=False)]


def compute_rampup_delta(df):
    initial = {}
    final = {}

    for seg in hold_segments(df, INITIAL_RAMP_STATE):
        target_kpa = float(seg["rampTarget_g"].iloc[0])
        stable = trim_stable_region(seg)
        stable = stable.dropna(subset=["compression_pct_smooth"])
        if not stable.empty:
            initial[target_kpa] = stable["compression_pct_smooth"].mean()

    for seg in hold_segments(df, FINAL_RAMP_STATE):
        target_kpa = float(seg["rampTarget_g"].iloc[0])
        stable = trim_stable_region(seg)
        stable = stable.dropna(subset=["compression_pct_smooth"])
        if not stable.empty:
            final[target_kpa] = stable["compression_pct_smooth"].mean()

    rows = []
    for target_kpa in sorted(set(initial) & set(final)):
        rows.append({
            "target_kpa": target_kpa,
            "delta_compression_pct": final[target_kpa] - initial[target_kpa],
        })
    return pd.DataFrame(rows)


def load_fatigue_delta(scale):
    path = scale["fatigue_dir"] / f"{scale['fatigue_prefix']}_ramp_hold_comparison_summary.csv"
    df = pd.read_csv(path)
    df["target_kpa"] = pd.to_numeric(df["target_kpa"], errors="coerce")
    df["delta_compression_pct"] = pd.to_numeric(df["delta_compression_pct"], errors="coerce")
    return df[["target_kpa", "delta_compression_pct"]].dropna().sort_values("target_kpa")


def plot_delta_comparison(rampup_deltas, fatigue_deltas):
    fig, axes = plt.subplots(
        1,
        len(SCALES),
        figsize=(9.0, 3.0),
        sharey=True,
        constrained_layout=True,
    )

    for ax, scale in zip(axes, SCALES):
        ramp_df = rampup_deltas.get(scale["label"])
        fat_df = fatigue_deltas.get(scale["label"])
        if ramp_df is None or ramp_df.empty or fat_df is None or fat_df.empty:
            ax.set_visible(False)
            continue

        ax.plot(
            ramp_df["target_kpa"],
            ramp_df["delta_compression_pct"],
            color=scale["color"],
            linewidth=2.2,
            label="Progressive delta",
            linestyle="-",
        )
        ax.plot(
            fat_df["target_kpa"],
            fat_df["delta_compression_pct"],
            color=scale["color"],
            linewidth=2.2,
            label="Fatigue delta",
            linestyle="--",
        )

        ax.text(
            0.02,
            0.96,
            scale["display"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
        )
        ax.set_xlabel("Ramp-step pressure [kPa]")
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_ylim(0, max(
            ramp_df["delta_compression_pct"].max(),
            fat_df["delta_compression_pct"].max()
        ) * 1.08)
        ax.grid(True, alpha=0.20)

    axes[0].set_ylabel("Compression increase [pp]")
    for ax in axes:
        ax.legend(frameon=False, loc="upper right")

    fig.savefig(FIGURE_FILE, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    rampup_deltas = {}
    fatigue_deltas = {}

    for scale in SCALES:
        df = load_rampup_data(scale["label"])
        rampup_deltas[scale["label"]] = compute_rampup_delta(df)
        fatigue_deltas[scale["label"]] = load_fatigue_delta(scale)

    plot_delta_comparison(rampup_deltas, fatigue_deltas)
    print(f"Wrote {FIGURE_FILE}")


if __name__ == "__main__":
    main()
