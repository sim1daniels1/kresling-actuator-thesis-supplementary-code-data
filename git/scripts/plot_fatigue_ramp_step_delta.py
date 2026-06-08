from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.dpi": 600,
})

BASE_DIR = Path(__file__).resolve().parent
SCALES = [
    {"label": "10x", "display": "10x", "dir": BASE_DIR / "10x", "color": "tab:blue"},
    {"label": "20x", "display": "20x", "dir": BASE_DIR / "20x_new", "color": "tab:green"},
    {"label": "40x", "display": "40x", "dir": BASE_DIR / "40x_new", "color": "tab:red"},
]

INITIAL_FILE_NAME = "aligned_initial_ramp.csv"
FINAL_FILE_NAME = "aligned_final_ramp.csv"
INITIAL_STATE = "RAMP_PRE"
FINAL_STATE = "RAMP_POST"
TARGETS_KPA = np.arange(0, 101, 10, dtype=float)

ZERO_KPA_MAX = 2.0
REF_FALLBACK_N = 20
SMOOTH_WINDOW = 5
EDGE_TRIM_FRAC = 0.15
EDGE_TRIM_MIN_POINTS = 2
MIN_STABLE_POINTS = 5
PRESSURE_DEADBAND_KPA = 2.0
OUTLIER_SIGMA_CUTOFF = 3.0
HOLD_START_TRIM_FRAC = 0.35
HOLD_START_TRIM_MIN_POINTS = 3

FIGURE_FILE = BASE_DIR / "fatigue_ramp_step_delta.png"


def rolling_mean(series, window):
    return series.rolling(window=window, center=True, min_periods=1).mean()


def robust_sigma(series):
    values = pd.Series(series).dropna()
    if values.empty:
        return np.nan
    median = values.median()
    mad = np.median(np.abs(values - median))
    return 1.4826 * mad


def keep_longest_contiguous_run(df):
    if df is None or df.empty:
        return df

    out = df.copy()
    out["orig_index"] = out.index
    new_run = out["orig_index"].diff().fillna(1).ne(1)
    out["run_id"] = new_run.cumsum()
    out = max((seg for _, seg in out.groupby("run_id", sort=False)), key=len).copy()
    return out.drop(columns=["orig_index", "run_id"], errors="ignore").reset_index(drop=True)


def load_ramp_csv(path):
    df = pd.read_csv(path)

    needed_cols = [
        "time_s",
        "state",
        "P_g",
        "P_g_smooth",
        "rampTarget_g",
        "rampHold",
        "Height_yellow_px",
        "Height_yellow_px_smooth",
    ]

    for col in needed_cols:
        if col != "state":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time_s", "state", "P_g_smooth", "rampTarget_g", "rampHold"]).reset_index(drop=True)
    return df


def stable_zero_kpa_reference_segment(df_init):
    zero_hold = df_init[
        (df_init["state"] == INITIAL_STATE)
        & (df_init["rampHold"] == 1)
        & np.isclose(df_init["rampTarget_g"], 0.0)
    ].copy()

    if zero_hold.empty:
        return None

    zero_hold["orig_index"] = zero_hold.index
    new_segment = zero_hold["orig_index"].diff().fillna(1).ne(1)
    zero_hold["segment_id"] = new_segment.cumsum()
    zero_hold = max((seg for _, seg in zero_hold.groupby("segment_id", sort=False)), key=len).copy()
    zero_hold = zero_hold.drop(columns=["orig_index", "segment_id"], errors="ignore")

    n_total = len(zero_hold)
    trim_n = max(EDGE_TRIM_MIN_POINTS, int(round(n_total * EDGE_TRIM_FRAC)))
    max_trim = max(0, (n_total - MIN_STABLE_POINTS) // 2)
    trim_n = min(trim_n, max_trim)
    stable = zero_hold.iloc[trim_n:n_total - trim_n].copy() if trim_n > 0 else zero_hold.copy()
    if len(stable) < MIN_STABLE_POINTS:
        stable = zero_hold.copy()

    n_stable = len(stable)
    start_trim_n = max(HOLD_START_TRIM_MIN_POINTS, int(round(n_stable * HOLD_START_TRIM_FRAC)))
    max_start_trim = max(0, n_stable - MIN_STABLE_POINTS)
    start_trim_n = min(start_trim_n, max_start_trim)
    if start_trim_n > 0:
        stable = stable.iloc[start_trim_n:].copy()

    pressure_mask = stable["P_g_smooth"].sub(0.0).abs() <= PRESSURE_DEADBAND_KPA
    if pressure_mask.sum() >= MIN_STABLE_POINTS:
        stable = stable.loc[pressure_mask].copy()
        stable = keep_longest_contiguous_run(stable)

    return stable.reset_index(drop=True)


def compute_reference_height(df_init):
    stable_zero = stable_zero_kpa_reference_segment(df_init)
    if stable_zero is not None and len(stable_zero) >= MIN_STABLE_POINTS:
        return stable_zero["Height_yellow_px_smooth"].median()

    zero_mask = df_init["P_g_smooth"] <= ZERO_KPA_MAX
    if zero_mask.sum() >= 3:
        return df_init.loc[zero_mask, "Height_yellow_px_smooth"].median()

    return df_init["Height_yellow_px_smooth"].iloc[:REF_FALLBACK_N].median()


def prepare_ramp(df, h_ref):
    out = df.copy()
    out["compression_pct"] = (h_ref - out["Height_yellow_px"]) / h_ref * 100.0
    out["compression_pct_smooth"] = rolling_mean(out["compression_pct"], SMOOTH_WINDOW)
    return out


def collect_hold_segments(df, ramp_state):
    hold_df = df[(df["state"] == ramp_state) & (df["rampHold"] == 1)].copy()
    if hold_df.empty:
        return []

    hold_df["orig_index"] = hold_df.index
    new_segment = (
        hold_df["rampTarget_g"].ne(hold_df["rampTarget_g"].shift())
        | hold_df["state"].ne(hold_df["state"].shift())
        | hold_df["orig_index"].diff().fillna(1).ne(1)
    )
    hold_df["segment_id"] = new_segment.cumsum()
    return [seg.reset_index(drop=True) for _, seg in hold_df.groupby("segment_id", sort=False)]


def choose_best_segment(segments, target_kpa):
    candidates = [seg for seg in segments if np.isclose(float(seg["rampTarget_g"].iloc[0]), target_kpa)]
    if not candidates:
        return None
    return max(candidates, key=len)


def trim_to_stable_region(seg_df, target_kpa):
    if seg_df is None or seg_df.empty:
        return None

    n_total = len(seg_df)
    trim_n = max(EDGE_TRIM_MIN_POINTS, int(round(n_total * EDGE_TRIM_FRAC)))
    max_trim = max(0, (n_total - MIN_STABLE_POINTS) // 2)
    trim_n = min(trim_n, max_trim)

    stable = seg_df.iloc[trim_n:n_total - trim_n].copy() if trim_n > 0 else seg_df.copy()
    if len(stable) < MIN_STABLE_POINTS:
        stable = seg_df.copy()

    n_stable = len(stable)
    start_trim_n = max(HOLD_START_TRIM_MIN_POINTS, int(round(n_stable * HOLD_START_TRIM_FRAC)))
    max_start_trim = max(0, n_stable - MIN_STABLE_POINTS)
    start_trim_n = min(start_trim_n, max_start_trim)
    if start_trim_n > 0:
        stable = stable.iloc[start_trim_n:].copy()

    pressure_error = stable["P_g_smooth"].sub(target_kpa).abs()
    pressure_mask = pressure_error <= PRESSURE_DEADBAND_KPA
    if pressure_mask.sum() >= MIN_STABLE_POINTS:
        stable = stable.loc[pressure_mask].copy()
        stable = keep_longest_contiguous_run(stable)

    comp_sigma = robust_sigma(stable["compression_pct_smooth"])
    comp_median = stable["compression_pct_smooth"].median()
    if np.isfinite(comp_sigma) and comp_sigma > 0:
        comp_mask = stable["compression_pct_smooth"].sub(comp_median).abs() <= OUTLIER_SIGMA_CUTOFF * comp_sigma
        if comp_mask.sum() >= MIN_STABLE_POINTS:
            stable = stable.loc[comp_mask].copy()
            stable = keep_longest_contiguous_run(stable)

    return stable.reset_index(drop=True)


def summarize_hold(stable_df, target_kpa, label):
    if stable_df is None or stable_df.empty:
        return {
            "ramp_label": label,
            "target_kpa": target_kpa,
            "n_points": 0,
            "time_start_s": np.nan,
            "time_end_s": np.nan,
            "pressure_mean_kpa": np.nan,
            "pressure_std_kpa": np.nan,
            "compression_mean_pct": np.nan,
            "compression_std_pct": np.nan,
        }

    return {
        "ramp_label": label,
        "target_kpa": target_kpa,
        "n_points": len(stable_df),
        "time_start_s": stable_df["time_s"].iloc[0],
        "time_end_s": stable_df["time_s"].iloc[-1],
        "pressure_mean_kpa": stable_df["P_g_smooth"].median(),
        "pressure_std_kpa": robust_sigma(stable_df["P_g_smooth"]),
        "compression_mean_pct": stable_df["compression_pct_smooth"].median(),
        "compression_std_pct": robust_sigma(stable_df["compression_pct_smooth"]),
    }


def build_step_summary(df, ramp_state, label):
    segments = collect_hold_segments(df, ramp_state)
    rows = []

    for target_kpa in TARGETS_KPA:
        best_segment = choose_best_segment(segments, target_kpa)
        stable_segment = trim_to_stable_region(best_segment, target_kpa)
        rows.append(summarize_hold(stable_segment, target_kpa, label))

    return pd.DataFrame(rows)


def combine_pre_post(init_summary, final_summary):
    init_part = init_summary.add_prefix("pre_")
    final_part = final_summary.add_prefix("post_")

    merged = init_part.merge(
        final_part,
        left_on="pre_target_kpa",
        right_on="post_target_kpa",
        how="outer",
    )

    merged["target_kpa"] = merged["pre_target_kpa"].combine_first(merged["post_target_kpa"])
    merged["delta_compression_pct"] = (
        merged["post_compression_mean_pct"] - merged["pre_compression_mean_pct"]
    )
    merged["delta_error_pct"] = np.sqrt(
        merged["pre_compression_std_pct"].fillna(0.0) ** 2
        + merged["post_compression_std_pct"].fillna(0.0) ** 2
    )

    return merged.sort_values("target_kpa").reset_index(drop=True)


def compute_fatigue_delta(scale_dir):
    initial_path = scale_dir / INITIAL_FILE_NAME
    final_path = scale_dir / FINAL_FILE_NAME
    if not initial_path.exists() or not final_path.exists():
        raise FileNotFoundError(f"Missing input files in {scale_dir}")

    init_df = load_ramp_csv(initial_path)
    final_df = load_ramp_csv(final_path)
    h_ref = compute_reference_height(init_df)

    init_df = prepare_ramp(init_df, h_ref)
    final_df = prepare_ramp(final_df, h_ref)

    init_summary = build_step_summary(init_df, INITIAL_STATE, "pre")
    final_summary = build_step_summary(final_df, FINAL_STATE, "post")
    return combine_pre_post(init_summary, final_summary)


def plot_fatigue_delta(scale_deltas):
    fig, axes = plt.subplots(
        1,
        len(SCALES),
        figsize=(9.0, 3.0),
        sharey=True,
        constrained_layout=True,
    )

    y_values = []
    axis_data = []

    for scale in SCALES:
        df = scale_deltas.get(scale["label"])
        if df is None or df.empty:
            axis_data.append(None)
            continue

        df = df.dropna(subset=["target_kpa", "delta_compression_pct"])
        if df.empty:
            axis_data.append(None)
            continue

        x = df["target_kpa"].to_numpy(dtype=float)
        y = df["delta_compression_pct"].to_numpy(dtype=float)
        y_values.append(y)
        axis_data.append((x, y, scale))

    if y_values:
        global_y = np.concatenate(y_values)
        y_min = max(0.0, global_y.min() - max(0.1, (global_y.max() - global_y.min()) * 0.12))
        y_max = global_y.max() + max(0.2, (global_y.max() - global_y.min()) * 0.12)
    else:
        y_min, y_max = 0.0, 1.0

    for ax, data in zip(axes, axis_data):
        if data is None:
            ax.set_visible(False)
            continue

        x, y, scale = data
        ax.plot(
            x,
            y,
            color=scale["color"],
            linewidth=2.2,
            marker="o",
            markersize=4,
        )
        ax.axhline(0.0, color="0.45", linewidth=1.0, alpha=0.8)
        # place specimen label at the way top-left of the panel (slightly above)
        ax.text(
            0.02,
            1.06,
            scale["display"],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

        ax.set_xlabel("Pressure [kPa]")
        ax.set_xlim(0, 100)
        # major and minor x ticks (major every 20 kPa, minor every 10 kPa)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(10))

        # denser y minor ticks (auto subdivisions) for consistent grid density
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.set_ylim(y_min, y_max)
        # major grid lighter, minor grid finer
        ax.grid(True, which="major", axis="both", alpha=0.16, linewidth=0.7)
        ax.grid(True, which="minor", axis="both", alpha=0.10, linewidth=0.4)

    axes[0].set_ylabel("Compression change [pp]")

    fig.savefig(FIGURE_FILE, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    scale_deltas = {}

    for scale in SCALES:
        try:
            df = compute_fatigue_delta(scale["dir"])
            scale_deltas[scale["label"]] = df
            print(f"{scale['display']}: computed delta for {len(df)} target pressures")
        except Exception as exc:
            print(f"{scale['display']}: failed to compute delta - {exc}")
            scale_deltas[scale["label"]] = None

    plot_fatigue_delta(scale_deltas)
    print(f"Wrote {FIGURE_FILE}")


if __name__ == "__main__":
    main()
