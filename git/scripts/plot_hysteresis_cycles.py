from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


BASE_DIR = Path(__file__).resolve().parent

TESTS = [
    {
        "label": "10x",
        "csv": BASE_DIR / "10x" / "10x_aligned_all.csv",
        "reference_csv": BASE_DIR / "10x" / "aligned_initial_ramp.csv",
        "out_dir": BASE_DIR / "10x",
        "prefix": "10x",
    },
    {
        "label": "20x_new",
        "csv": BASE_DIR / "20x_new" / "20x_aligned_all.csv",
        "reference_csv": BASE_DIR / "20x_new" / "aligned_initial_ramp.csv",
        "out_dir": BASE_DIR / "20x_new",
        "prefix": "20x_new",
    },
    {
        "label": "40x_new",
        "csv": BASE_DIR / "40x_new" / "40x_aligned_all.csv",
        "reference_csv": BASE_DIR / "40x_new" / "aligned_initial_ramp.csv",
        "out_dir": BASE_DIR / "40x_new",
        "prefix": "40x_new",
    },
]

CYCLIC_STATES = {"UP", "HOLD_HIGH", "DOWN", "HOLD_LOW"}
INITIAL_STATE = "RAMP_PRE"
SELECTED_CYCLES = [100, 200, 300, 400, 500]
MAX_CYCLES = 500

HEIGHT_COLS = ["Height_yellow_px_smooth", "Height_yellow_px"]
PRESSURE_COLS = ["P_g_smooth", "P_g"]

ZERO_KPA_MAX = 2.0
REF_FALLBACK_N = 20
EDGE_TRIM_FRAC = 0.15
EDGE_TRIM_MIN_POINTS = 2
MIN_STABLE_POINTS = 5
PRESSURE_DEADBAND_KPA = 2.0
HOLD_START_TRIM_FRAC = 0.35
HOLD_START_TRIM_MIN_POINTS = 3


def first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of these columns were found: {candidates}")


def numeric(df, columns):
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def keep_longest_contiguous_run(df):
    if df is None or df.empty:
        return df

    out = df.copy()
    out["orig_index"] = out.index
    new_run = out["orig_index"].diff().fillna(1).ne(1)
    out["run_id"] = new_run.cumsum()
    out = max((seg for _, seg in out.groupby("run_id", sort=False)), key=len).copy()
    return out.drop(columns=["orig_index", "run_id"], errors="ignore").reset_index(drop=True)


def stable_zero_kpa_reference_segment(df, height_col, pressure_col):
    zero_hold = df[
        (df["state"] == INITIAL_STATE)
        & (df["rampHold"] == 1)
        & np.isclose(df["rampTarget_g"], 0.0)
        & df[height_col].notna()
    ].copy()

    if zero_hold.empty:
        return None

    zero_hold = keep_longest_contiguous_run(zero_hold)
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

    pressure_mask = stable[pressure_col].sub(0.0).abs() <= PRESSURE_DEADBAND_KPA
    if pressure_mask.sum() >= MIN_STABLE_POINTS:
        stable = stable.loc[pressure_mask].copy()
        stable = keep_longest_contiguous_run(stable)

    return stable.reset_index(drop=True)


def compute_reference_height(df, height_col, pressure_col):
    stable_zero = stable_zero_kpa_reference_segment(df, height_col, pressure_col)
    if stable_zero is not None and len(stable_zero) >= MIN_STABLE_POINTS:
        return float(stable_zero[height_col].median()), len(stable_zero), "initial 0 kPa hold"

    initial_ramp = df[(df["state"] == INITIAL_STATE) & df[height_col].notna()].copy()
    zero_mask = initial_ramp[pressure_col].abs() <= ZERO_KPA_MAX
    if zero_mask.sum() >= 3:
        zero_rows = initial_ramp.loc[zero_mask].copy()
        return float(zero_rows[height_col].median()), int(zero_mask.sum()), "initial near-0 kPa ramp rows"

    fallback = initial_ramp[height_col].dropna().iloc[:REF_FALLBACK_N]
    if fallback.empty:
        raise ValueError("Could not find initial ramp height rows for reference height.")
    return float(fallback.median()), len(fallback), "initial ramp fallback rows"


def load_cyclic_data(csv_path, reference_csv_path):
    df = pd.read_csv(csv_path)
    height_col = first_existing_column(df, HEIGHT_COLS)
    pressure_col = first_existing_column(df, PRESSURE_COLS)

    required = ["cycles", "state", "rampHold", "rampTarget_g", height_col, pressure_col]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {missing}")

    df = numeric(df, ["cycles", "rampHold", "rampTarget_g", height_col, pressure_col])
    df["state"] = df["state"].astype(str)

    reference_df = pd.read_csv(reference_csv_path) if reference_csv_path.exists() else df.copy()
    reference_height_col = first_existing_column(reference_df, HEIGHT_COLS)
    reference_pressure_col = first_existing_column(reference_df, PRESSURE_COLS)
    reference_required = [
        "state",
        "rampHold",
        "rampTarget_g",
        reference_height_col,
        reference_pressure_col,
    ]
    missing_reference = sorted(set(reference_required) - set(reference_df.columns))
    if missing_reference:
        raise ValueError(
            f"{reference_csv_path.name} is missing required columns: {missing_reference}"
        )
    reference_df = numeric(
        reference_df,
        ["rampHold", "rampTarget_g", reference_height_col, reference_pressure_col],
    )
    reference_df["state"] = reference_df["state"].astype(str)

    h_ref, h_ref_rows, h_ref_source = compute_reference_height(
        reference_df,
        reference_height_col,
        reference_pressure_col,
    )
    h_ref_source = f"{h_ref_source} in {reference_csv_path.name}"
    df["deformation_pct"] = (h_ref - df[height_col]) / h_ref * 100.0

    df = df[df["state"].isin(CYCLIC_STATES)].copy()
    df = df.dropna(subset=["cycles", "deformation_pct", pressure_col])

    # The controller log stores completed cycles. During the first physical
    # hysteresis loop cycles == 0, so plot as cycle_number == cycles + 1.
    df["cycle_number"] = df["cycles"].astype(int) + 1
    df = df[df["cycle_number"].between(1, MAX_CYCLES)].copy()

    df = df.rename(columns={pressure_col: "pressure"})
    out = df[["cycle_number", "deformation_pct", "pressure"]].reset_index(drop=True)
    return out, h_ref, h_ref_rows, h_ref_source


def set_common_axes(ax, title):
    ax.set_title(title)
    ax.set_xlabel("Deformation [%]")
    ax.set_ylabel("Gauge pressure, P_g [kPa]")
    ax.grid(True, alpha=0.25)


def plot_full_evolution(df, label, output_path):
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    segments = []
    colors = []
    for cycle, seg in df.groupby("cycle_number", sort=True):
        if len(seg) < 2:
            continue
        points = seg[["deformation_pct", "pressure"]].to_numpy(dtype=float)
        segments.append(np.column_stack([points[:-1], points[1:]]).reshape(-1, 2, 2))
        colors.extend([cycle] * (len(points) - 1))

    if not segments:
        raise ValueError(f"No plottable cyclic segments found for {label}.")

    line_segments = np.concatenate(segments, axis=0)
    collection = LineCollection(
        line_segments,
        array=np.asarray(colors, dtype=float),
        cmap="viridis",
        linewidths=0.7,
        alpha=0.55,
    )
    ax.add_collection(collection)
    ax.autoscale()

    cbar = fig.colorbar(collection, ax=ax, pad=0.02)
    cbar.set_label("Cycle number")
    set_common_axes(ax, f"{label} fatigue hysteresis evolution, cycles 1-500")

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def closed_cycle_points(seg):
    points = seg[["deformation_pct", "pressure"]].to_numpy(dtype=float)
    if len(points) < 2:
        return points
    return np.vstack([points, points[0]])


def plot_selected_cycles(df, label, output_path):
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(SELECTED_CYCLES)))

    plotted = []
    for target_cycle, color in zip(SELECTED_CYCLES, colors):
        seg = df[df["cycle_number"] == target_cycle].copy()
        if len(seg) < 2:
            continue
        points = closed_cycle_points(seg)
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=1.6,
            label=f"Cycle {target_cycle}",
        )
        plotted.append(target_cycle)

    if not plotted:
        raise ValueError(f"No selected cycles were available for {label}.")

    set_common_axes(ax, f"{label} fatigue hysteresis, selected cycles")
    ax.legend(title="Selected cycles", frameon=False, ncols=2)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return plotted


def main():
    for test in TESTS:
        df, h_ref, h_ref_rows, h_ref_source = load_cyclic_data(
            test["csv"],
            test["reference_csv"],
        )
        full_png = test["out_dir"] / f"{test['prefix']}_hysteresis_evolution_500cycles.png"
        selected_png = test["out_dir"] / f"{test['prefix']}_hysteresis_selected_cycles.png"

        plot_full_evolution(df, test["label"], full_png)
        plotted = plot_selected_cycles(df, test["label"], selected_png)

        cycle_min = int(df["cycle_number"].min())
        cycle_max = int(df["cycle_number"].max())
        print(f"{test['label']}: cycles {cycle_min}-{cycle_max}, rows {len(df)}")
        print(f"  reference height: {h_ref:.3f} px from {h_ref_rows} {h_ref_source}")
        print(f"  wrote {full_png}")
        print(f"  wrote {selected_png}")
        print(f"  selected cycles plotted: {plotted}")


if __name__ == "__main__":
    main()
