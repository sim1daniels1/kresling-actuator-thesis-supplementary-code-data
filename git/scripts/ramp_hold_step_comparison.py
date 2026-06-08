import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# SETTINGS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
CYCLE_LABEL = BASE_DIR.name

INITIAL_FILE = BASE_DIR / "aligned_initial_ramp.csv"
FINAL_FILE = BASE_DIR / "aligned_final_ramp.csv"

INITIAL_STATE = "RAMP_PRE"
FINAL_STATE = "RAMP_POST"
TARGETS_KPA = np.arange(0, 101, 10, dtype=float)

# Reference height for 0% compression from the stable 0 kPa hold in the initial ramp
ZERO_KPA_MAX = 2.0
REF_FALLBACK_N = 20

# Smoothing and hold extraction
SMOOTH_WINDOW = 5
EDGE_TRIM_FRAC = 0.15
EDGE_TRIM_MIN_POINTS = 2
MIN_STABLE_POINTS = 5
PRESSURE_DEADBAND_KPA = 2.0
OUTLIER_SIGMA_CUTOFF = 3.0
HOLD_START_TRIM_FRAC = 0.35
HOLD_START_TRIM_MIN_POINTS = 3

# Outputs
SUMMARY_CSV = BASE_DIR / f"{CYCLE_LABEL}_ramp_hold_comparison_summary.csv"
FIGURE_FILE = BASE_DIR / f"{CYCLE_LABEL}_ramp_hold_comparison.png"
# =========================================================


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

    segments = []
    for _, seg in hold_df.groupby("segment_id", sort=False):
        segments.append(seg.reset_index(drop=True))

    return segments


def choose_best_segment(segments, target_kpa):
    candidates = []

    for seg in segments:
        seg_target = float(seg["rampTarget_g"].iloc[0])
        if np.isclose(seg_target, target_kpa):
            candidates.append(seg)

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

    # Simple propagated spread for visual guidance on delta bars
    merged["delta_error_pct"] = np.sqrt(
        merged["pre_compression_std_pct"].fillna(0.0) ** 2
        + merged["post_compression_std_pct"].fillna(0.0) ** 2
    )

    merged = merged.sort_values("target_kpa").reset_index(drop=True)
    return merged


def plot_step_comparison(summary_df):
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.4]},
    )

    x = summary_df["target_kpa"].to_numpy(dtype=float)

    pre_mean = summary_df["pre_compression_mean_pct"].to_numpy(dtype=float)
    post_mean = summary_df["post_compression_mean_pct"].to_numpy(dtype=float)

    delta = summary_df["delta_compression_pct"].to_numpy(dtype=float)

    ax_top.plot(
        x,
        pre_mean,
        "o-",
        linewidth=2,
        label="Initial ramp",
    )
    ax_top.plot(
        x,
        post_mean,
        "o-",
        linewidth=2,
        label="Final ramp",
    )
    ax_top.set_ylabel("Axial compression [%]")
    ax_top.set_title(f"{CYCLE_LABEL} fatigue: ramp-step hold comparison")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="best")

    bar_colors = np.where(delta >= 0.0, "tab:orange", "tab:blue")
    ax_bottom.bar(
        x,
        delta,
        width=7.0,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
    )
    ax_bottom.axhline(0.0, color="black", linewidth=1.0)
    ax_bottom.set_xlabel("Ramp target pressure [kPa]")
    ax_bottom.set_ylabel("Increase in compression after fatigue [%]")
    ax_bottom.grid(True, axis="y", alpha=0.3)
    ax_bottom.set_xticks(TARGETS_KPA)

    ax_bottom.text(
        0.98,
        0.96,
        "0 kPa = residual compression",
        transform=ax_bottom.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )

    plt.tight_layout()
    return fig


def main():
    init_df = load_ramp_csv(INITIAL_FILE)
    final_df = load_ramp_csv(FINAL_FILE)

    h_ref = compute_reference_height(init_df)
    print(f"Reference height from initial ramp at ~0 kPa: {h_ref:.3f} px")

    init_df = prepare_ramp(init_df, h_ref)
    final_df = prepare_ramp(final_df, h_ref)

    init_summary = build_step_summary(init_df, INITIAL_STATE, "pre")
    final_summary = build_step_summary(final_df, FINAL_STATE, "post")
    summary_df = combine_pre_post(init_summary, final_summary)

    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary table: {SUMMARY_CSV.name}")
    print(
        summary_df[
            [
                "target_kpa",
                "pre_compression_mean_pct",
                "post_compression_mean_pct",
                "delta_compression_pct",
                "pre_pressure_mean_kpa",
                "post_pressure_mean_kpa",
                "pre_n_points",
                "post_n_points",
            ]
        ].round(3).to_string(index=False)
    )

    fig = plot_step_comparison(summary_df)
    fig.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {FIGURE_FILE.name}")
    plt.show()


if __name__ == "__main__":
    main()
