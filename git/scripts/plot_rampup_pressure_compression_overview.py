from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FATIGUE_DIR = PROJECT_DIR / "fatigue"

FIGURE_FILE = BASE_DIR / "rampup_pressure_compression_overview.png"
AVERAGES_CSV = BASE_DIR / "rampup_pressure_compression_hold_averages.csv"

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

    return df, path


def load_ramp_step_summary(scale):
    path = scale["fatigue_dir"] / f"{scale['fatigue_prefix']}_ramp_hold_comparison_summary.csv"
    df = pd.read_csv(path)

    required = [
        "pre_pressure_mean_kpa",
        "pre_compression_mean_pct",
        "post_pressure_mean_kpa",
        "post_compression_mean_pct",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
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


def rampup_hold_averages(df, scale):
    rows = []
    segments_by_target = {}
    for seg in hold_segments(df, INITIAL_RAMP_STATE):
        target_kpa = float(seg["rampTarget_g"].iloc[0])
        segments_by_target.setdefault(target_kpa, []).append(seg)

    for target_kpa in sorted(segments_by_target):
        seg = max(segments_by_target[target_kpa], key=len)
        stable = trim_stable_region(seg)
        stable = stable.dropna(subset=["P_g_smooth", "compression_pct_smooth"])
        if stable.empty:
            continue

        rows.append(
            {
                "scale": scale["label"],
                "display": scale["display"],
                "target_kpa": target_kpa,
                "n_points": len(stable),
                "pressure_mean_kpa": stable["P_g_smooth"].mean(),
                "compression_mean_pct": stable["compression_pct_smooth"].mean(),
                "pressure_std_kpa": stable["P_g_smooth"].std(ddof=0),
                "compression_std_pct": stable["compression_pct_smooth"].std(ddof=0),
            }
        )

    return pd.DataFrame(rows)


def clean_xy(df, x_col, y_col):
    return df.dropna(subset=[x_col, y_col]).copy()


def style_pressure_compression_axis(ax):
    ax.set_xlabel("Pressure [kPa]")
    ax.set_ylabel("Compression [%]")
    ax.grid(True, alpha=0.25)


def plot_overview(raw_data, averages, ramp_step_summaries):
    fig = plt.figure(figsize=(13.5, 14.5), constrained_layout=True)
    grid = fig.add_gridspec(
        3,
        6,
        height_ratios=[1.15, 1.0, 1.0],
    )

    ax_average = fig.add_subplot(grid[0, 0:3])
    ax_placeholder = fig.add_subplot(grid[0, 3:6])
    raw_axes = [fig.add_subplot(grid[1, i * 2:(i + 1) * 2]) for i in range(3)]
    step_axes = [fig.add_subplot(grid[2, i * 2:(i + 1) * 2]) for i in range(3)]

    for ax in [ax_average, ax_placeholder, *raw_axes, *step_axes]:
        ax.set_box_aspect(1)

    for scale in SCALES:
        scale_avg = averages[averages["scale"] == scale["label"]].sort_values("pressure_mean_kpa")
        ax_average.plot(
            scale_avg["pressure_mean_kpa"],
            scale_avg["compression_mean_pct"],
            color=scale["color"],
            linewidth=2.1,
            marker="o",
            markersize=3.8,
            label=scale["display"],
        )

    style_pressure_compression_axis(ax_average)
    ax_average.set_title("Ramp-up hold averages", loc="left", fontweight="bold")
    ax_average.legend(frameon=False, loc="best")

    ax_placeholder.set_xticks([])
    ax_placeholder.set_yticks([])
    ax_placeholder.text(
        0.5,
        0.5,
        "Placeholder",
        ha="center",
        va="center",
        transform=ax_placeholder.transAxes,
        color="0.35",
    )
    ax_placeholder.set_title("Placeholder", loc="left", fontweight="bold")
    for spine in ax_placeholder.spines.values():
        spine.set_color("0.75")

    for ax, scale in zip(raw_axes, SCALES):
        df = raw_data[scale["label"]]
        pre = clean_xy(df[df["state"] == INITIAL_RAMP_STATE], "P_g", "compression_pct")
        post = clean_xy(df[df["state"] == FINAL_RAMP_STATE], "P_g", "compression_pct")

        ax.plot(
            pre["P_g"],
            pre["compression_pct"],
            color=scale["color"],
            linewidth=1.4,
            alpha=0.65,
            label="Initial ramp",
        )
        ax.plot(
            post["P_g"],
            post["compression_pct"],
            color=scale["color"],
            linewidth=1.4,
            alpha=0.65,
            linestyle="--",
            label="Final ramp",
        )
        ax.set_title(f"{scale['display']} raw ramp-up", loc="left", fontweight="bold")
        style_pressure_compression_axis(ax)

    raw_axes[0].legend(frameon=False, loc="best")

    for ax, scale in zip(step_axes, SCALES):
        summary = ramp_step_summaries[scale["label"]]
        pre = clean_xy(summary, "pre_pressure_mean_kpa", "pre_compression_mean_pct")
        post = clean_xy(summary, "post_pressure_mean_kpa", "post_compression_mean_pct")

        ax.plot(
            pre["pre_pressure_mean_kpa"],
            pre["pre_compression_mean_pct"],
            color=scale["color"],
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label="Before cycling",
        )
        ax.plot(
            post["post_pressure_mean_kpa"],
            post["post_compression_mean_pct"],
            color=scale["color"],
            linewidth=2.0,
            linestyle="--",
            marker="o",
            markersize=3.5,
            label="After cycling",
        )
        ax.set_title(f"{scale['display']} ramp-step", loc="left", fontweight="bold")
        style_pressure_compression_axis(ax)

    step_axes[0].legend(frameon=False, loc="best")

    fig.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    raw_data = {}
    average_tables = []
    ramp_step_summaries = {}

    for scale in SCALES:
        df, path = load_rampup_data(scale["label"])
        raw_data[scale["label"]] = df
        average_tables.append(rampup_hold_averages(df, scale))
        ramp_step_summaries[scale["label"]] = load_ramp_step_summary(scale)
        print(f"{scale['display']}: loaded {path.relative_to(BASE_DIR)} ({len(df)} rows)")

    averages = pd.concat(average_tables, ignore_index=True)
    averages.to_csv(AVERAGES_CSV, index=False)
    plot_overview(raw_data, averages, ramp_step_summaries)

    print(f"Wrote {AVERAGES_CSV}")
    print(f"Wrote {FIGURE_FILE}")


if __name__ == "__main__":
    main()
