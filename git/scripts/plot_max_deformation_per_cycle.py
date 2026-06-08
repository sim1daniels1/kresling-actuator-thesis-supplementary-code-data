from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_hysteresis_cycles import (
    HEIGHT_COLS,
    MAX_CYCLES,
    PRESSURE_COLS,
    TESTS,
    compute_reference_height,
    first_existing_column,
    numeric,
)


BASE_DIR = Path(__file__).resolve().parent

HIGH_PRESSURE_STATE = "HOLD_HIGH"
LOW_PRESSURE_STATE = "HOLD_LOW"
HIGH_PRESSURE_MIN_KPA = 90.0
LOW_PRESSURE_MAX_KPA = 10.0
HIGH_PRESSURE_FALLBACK_BAND_KPA = 5.0
LOW_PRESSURE_FALLBACK_BAND_KPA = 5.0
ROLLING_WINDOW_CYCLES = 15

SUMMARY_CSV = BASE_DIR / "deformation_extrema_per_cycle_summary.csv"
COMBINED_FIGURE = BASE_DIR / "deformation_extrema_per_cycle_evolution.png"
COMBINED_CHANGE_FIGURE = BASE_DIR / "deformation_extrema_change_from_initial.png"

PLOT_ORDER = ["10x", "20x_new", "40x_new"]
DISPLAY_LABELS = {
    "10x": "10x",
    "20x_new": "20x",
    "40x_new": "40x",
}
COLORS = {
    "10x": "tab:blue",
    "20x_new": "tab:green",
    "40x_new": "tab:red",
}


def ramp_step_summary_path(test_label):
    test = next(item for item in TESTS if item["label"] == test_label)
    return test["out_dir"] / f"{test['prefix']}_ramp_hold_comparison_summary.csv"


def load_ramp_step_summary(test_label):
    path = ramp_step_summary_path(test_label)
    if not path.exists():
        return None

    df = pd.read_csv(path)
    required = [
        "target_kpa",
        "pre_compression_mean_pct",
        "post_compression_mean_pct",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    return numeric(df, required).dropna(subset=required).copy()


def reference_height_for_test(test):
    reference_df = pd.read_csv(test["reference_csv"])
    height_col = first_existing_column(reference_df, HEIGHT_COLS)
    pressure_col = first_existing_column(reference_df, PRESSURE_COLS)
    reference_df = numeric(
        reference_df,
        ["rampHold", "rampTarget_g", height_col, pressure_col],
    )
    reference_df["state"] = reference_df["state"].astype(str)
    h_ref, h_ref_rows, h_ref_source = compute_reference_height(
        reference_df,
        height_col,
        pressure_col,
    )
    return h_ref, h_ref_rows, h_ref_source


def load_deformation_data(test):
    df = pd.read_csv(test["csv"])
    height_col = first_existing_column(df, HEIGHT_COLS)
    pressure_col = first_existing_column(df, PRESSURE_COLS)

    required = ["cycles", "state", height_col, pressure_col]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{test['csv'].name} is missing required columns: {missing}")

    df = numeric(df, ["cycles", height_col, pressure_col])
    df["state"] = df["state"].astype(str)

    h_ref, h_ref_rows, h_ref_source = reference_height_for_test(test)
    df["deformation_pct"] = (h_ref - df[height_col]) / h_ref * 100.0
    df["pressure"] = df[pressure_col]

    df = df.dropna(subset=["cycles", "pressure", "deformation_pct"]).copy()
    df["cycle_number"] = df["cycles"].astype(int) + 1
    df = df[df["cycle_number"].between(1, MAX_CYCLES)].copy()

    return df, h_ref, h_ref_rows, h_ref_source


def high_pressure_rows_for_cycle(cycle_df):
    high = cycle_df[
        (cycle_df["state"] == HIGH_PRESSURE_STATE)
        & (cycle_df["pressure"] >= HIGH_PRESSURE_MIN_KPA)
    ].copy()
    if not high.empty:
        return high, f"{HIGH_PRESSURE_STATE} >= {HIGH_PRESSURE_MIN_KPA:g} kPa"

    max_pressure = cycle_df["pressure"].max()
    high = cycle_df[cycle_df["pressure"] >= max_pressure - HIGH_PRESSURE_FALLBACK_BAND_KPA].copy()
    return high, f"within {HIGH_PRESSURE_FALLBACK_BAND_KPA:g} kPa of cycle max pressure"


def low_pressure_rows_for_cycle(cycle_df):
    low = cycle_df[
        (cycle_df["state"] == LOW_PRESSURE_STATE)
        & (cycle_df["pressure"] <= LOW_PRESSURE_MAX_KPA)
    ].copy()
    if not low.empty:
        return low, f"{LOW_PRESSURE_STATE} <= {LOW_PRESSURE_MAX_KPA:g} kPa"

    min_pressure = cycle_df["pressure"].min()
    low = cycle_df[cycle_df["pressure"] <= min_pressure + LOW_PRESSURE_FALLBACK_BAND_KPA].copy()
    return low, f"within {LOW_PRESSURE_FALLBACK_BAND_KPA:g} kPa of cycle min pressure"


def summarize_deformation_extrema(test):
    df, h_ref, h_ref_rows, h_ref_source = load_deformation_data(test)

    rows = []
    for cycle_number, cycle_df in df.groupby("cycle_number", sort=True):
        high, selection = high_pressure_rows_for_cycle(cycle_df)
        low, low_selection = low_pressure_rows_for_cycle(cycle_df)
        if high.empty or low.empty:
            continue

        idx_max = high["deformation_pct"].idxmax()
        idx_min = low["deformation_pct"].idxmin()
        rows.append(
            {
                "test": test["label"],
                "cycle_number": int(cycle_number),
                "max_deformation_pct": float(high.loc[idx_max, "deformation_pct"]),
                "min_deformation_pct": float(low.loc[idx_min, "deformation_pct"]),
                "deformation_amplitude_pct": float(
                    high.loc[idx_max, "deformation_pct"] - low.loc[idx_min, "deformation_pct"]
                ),
                "pressure_at_max_deformation_kpa": float(high.loc[idx_max, "pressure"]),
                "pressure_at_min_deformation_kpa": float(low.loc[idx_min, "pressure"]),
                "mean_high_pressure_kpa": float(high["pressure"].mean()),
                "mean_low_pressure_kpa": float(low["pressure"].mean()),
                "max_high_pressure_kpa": float(high["pressure"].max()),
                "min_low_pressure_kpa": float(low["pressure"].min()),
                "n_high_pressure_points": int(len(high)),
                "n_low_pressure_points": int(len(low)),
                "high_pressure_selection": selection,
                "low_pressure_selection": low_selection,
                "reference_height_px": float(h_ref),
                "reference_rows": int(h_ref_rows),
                "reference_source": h_ref_source,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No cycle extrema summaries could be made for {test['label']}.")

    for col in ["max_deformation_pct", "min_deformation_pct", "deformation_amplitude_pct"]:
        out[f"{col}_smooth"] = out[col].rolling(
            ROLLING_WINDOW_CYCLES,
            center=True,
            min_periods=1,
        ).mean()

    return out

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "savefig.dpi": 600,
    "axes.xmargin": 0.05,
    "axes.ymargin": 0.05,
})


def plot_combined(summary):


# fig, ax = plt.subplots(
#     1, 3,
#     figsize=(6.7, 2.5),
#     sharey=True,
#     sharex=True,
#     layout="constrained",
# )




    fig, axes = plt.subplots(
        2,
        len(PLOT_ORDER),
        figsize=(6.7, 4),


        sharey="row",
        gridspec_kw={"height_ratios": [1.15, 1.0]},
        
        layout="constrained",
    )


    fig.set_constrained_layout_pads(
        w_pad=0.01,
        h_pad=0.01,
        wspace=0.02,
        hspace=0.02,
    )


    for ax, test_label in zip(axes[0], PLOT_ORDER):
        group = summary[summary["test"] == test_label].copy()
        if group.empty:
            ax.set_visible(False)
            continue

        color = COLORS.get(test_label)
        display_label = DISPLAY_LABELS.get(test_label, test_label)
        ax.fill_between(
            group["cycle_number"],
            group["min_deformation_pct_smooth"],
            group["max_deformation_pct_smooth"],
            color=color,
            alpha=0.12,
            linewidth=0,
        )

        # ax.plot(
        #     group["cycle_number"],
        #     group["max_deformation_pct"],
        #     color=color,
        #     alpha=0.22,
        #     linewidth=0.8,
        # )
        ax.plot(
            group["cycle_number"],
            group["max_deformation_pct_smooth"],
            color=color,
            linewidth=2.2,
            linestyle="-",
            label="Max compression",
        )
        # ax.plot(
        #     group["cycle_number"],
        #     group["min_deformation_pct"],
        #     color=color,
        #     alpha=0.22,
        #     linewidth=0.8,
        # )
        ax.plot(
            group["cycle_number"],
            group["min_deformation_pct_smooth"],
            color=color,
            linewidth=2.2,
            linestyle="--",
            label="Min compression",
        )

        # ax.set_title(display_label, loc="left", fontweight="bold")
        ax.text(
        0.02,
        0.98,
        display_label,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontweight="bold",

    )
        
        if ax == axes[1, 0] or ax == axes[0, 0]:
            ax.set_ylabel("Compression [%]")
        if ax == axes[0,1]:
            ax.set_xlabel("Cycle number")
            

            
        ax.set_xlim(1, MAX_CYCLES)
        ax.grid(True, alpha=0.25)

    compression_legend_handles = [
        Line2D([0], [0], color="0.25", linestyle="-", linewidth=2.2, label="Max compression"),
        Line2D([0], [0], color="0.25", linestyle="--", linewidth=2.2, label="Min compression"),
    ]
    axes[0, 0].legend(
        handles=compression_legend_handles,
        frameon=False,
        loc="best",
        handlelength=3.2,
        handletextpad=0.8,
    )

    for ax, test_label in zip(axes[1], PLOT_ORDER):
        ramp_summary = load_ramp_step_summary(test_label)
        if ramp_summary is None or ramp_summary.empty:
            ax.set_visible(False)
            continue

        color = COLORS.get(test_label)
        ax.plot(
            ramp_summary["target_kpa"],
            ramp_summary["pre_compression_mean_pct"],
            color=color,
            linewidth=2.2,
            linestyle="-",
            # marker="o",

            markersize=3.5,
            label="Pre cycling",
        )
        ax.plot(
            ramp_summary["target_kpa"],
            ramp_summary["post_compression_mean_pct"],
            color=color,
            linewidth=2.2,
            linestyle="--",
            # marker="o",
            markersize=3.5,
            label="Post cycling",
        )
        if ax == axes[1, 1]:
            ax.set_xlabel("Ramp-step pressure [kPa]")
        if ax == axes[1, 0] or ax == axes[0, 0]:
            ax.set_ylabel("Compression [%]")

        ax.set_xlim(ramp_summary["target_kpa"].min(), ramp_summary["target_kpa"].max())
        ax.grid(True, alpha=0.25)

    ramp_legend_handles = [
        Line2D([0], [0], color="0.25", linestyle="-", linewidth=2.2, label="Pre cycling"),
        Line2D([0], [0], color="0.25", linestyle="--", linewidth=2.2, label="Post cycling"),
    ]
    axes[1, 0].legend(
        handles=ramp_legend_handles,
        frameon=False,
        loc="upper left",

        handlelength=3.2,
        handletextpad=0.8,
    )
    fig.savefig(COMBINED_FIGURE, dpi=600)
    plt.close(fig)


def plot_combined_change(summary):
    fig, ax = plt.subplots(figsize=(9, 5.8), constrained_layout=True)

    for test_label, group in summary.groupby("test", sort=False):
        color = COLORS.get(test_label)
        display_label = DISPLAY_LABELS.get(test_label, test_label)
        baseline = group[group["cycle_number"].between(1, 25)]
        max_ref = baseline["max_deformation_pct_smooth"].mean()
        min_ref = baseline["min_deformation_pct_smooth"].mean()

        ax.plot(
            group["cycle_number"],
            group["max_deformation_pct_smooth"] - max_ref,
            color=color,
            linewidth=2.2,
            linestyle="-",
            label=f"{display_label} max",
        )
        ax.plot(
            group["cycle_number"],
            group["min_deformation_pct_smooth"] - min_ref,
            color=color,
            linewidth=2.2,
            linestyle="--",
            label=f"{display_label} min",
        )

    ax.axhline(0.0, color="0.2", linewidth=0.9, alpha=0.6)
    ax.set_title("Change in compression extrema during cyclic fatigue")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Change from first 25 cycles [percentage points]")
    ax.set_xlim(1, MAX_CYCLES)
    ax.grid(True, alpha=0.25)
    ax.legend(title=f"{ROLLING_WINDOW_CYCLES}-cycle mean", frameon=False, ncols=2)

    fig.savefig(COMBINED_CHANGE_FIGURE, dpi=300)
    plt.close(fig)


def plot_individual(summary):
    for test_label, group in summary.groupby("test", sort=False):
        out_dir = next(test["out_dir"] for test in TESTS if test["label"] == test_label)
        prefix = next(test["prefix"] for test in TESTS if test["label"] == test_label)
        figure = out_dir / f"{prefix}_deformation_extrema_per_cycle.png"
        display_label = DISPLAY_LABELS.get(test_label, test_label)

        fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
        ax.fill_between(
            group["cycle_number"],
            group["min_deformation_pct_smooth"],
            group["max_deformation_pct_smooth"],
            color="tab:blue",
            alpha=0.18,
            label="Cyclic compression range",
        )
        ax.plot(
            group["cycle_number"],
            group["max_deformation_pct_smooth"],
            color="tab:blue",
            linewidth=2.2,
            label="Max compression",
        )
        ax.plot(
            group["cycle_number"],
            group["min_deformation_pct_smooth"],
            color="tab:orange",
            linewidth=2.2,
            label="Min compression",
        )
        ax.set_title(f"{display_label} compression envelope evolution")
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("Compression [%]")
        ax.set_xlim(1, MAX_CYCLES)
        ax.grid(True, alpha=0.25)
        ax.legend(title=f"{ROLLING_WINDOW_CYCLES}-cycle mean", frameon=False)

        fig.savefig(figure, dpi=300)
        plt.close(fig)
        print(f"  wrote {figure}")


def main():
    summaries = []
    for test in TESTS:
        summary = summarize_deformation_extrema(test)
        summaries.append(summary)

        h_ref = summary["reference_height_px"].iloc[0]
        h_ref_rows = summary["reference_rows"].iloc[0]
        h_ref_source = summary["reference_source"].iloc[0]
        cycle_min = int(summary["cycle_number"].min())
        cycle_max = int(summary["cycle_number"].max())
        print(f"{test['label']}: cycles {cycle_min}-{cycle_max}, rows {len(summary)}")
        print(f"  reference height: {h_ref:.3f} px from {h_ref_rows} {h_ref_source}")

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(SUMMARY_CSV, index=False)
    plot_combined(combined)
    plot_combined_change(combined)
    plot_individual(combined)

    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {COMBINED_FIGURE}")
    print(f"Wrote {COMBINED_CHANGE_FIGURE}")


if __name__ == "__main__":
    main()
