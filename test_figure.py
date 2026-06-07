import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 450,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.2,
            "lines.solid_capstyle": "round",
            "grid.alpha": 0.25,
        }
    )


def ramp_sequence(start_x, start_p=0, end_p=100, step=10, dx=0.35, initial_plateau=True):
    xs, ys = [start_x], [start_p]
    x = start_x
    current_p = start_p

    direction = 1 if end_p >= start_p else -1
    first_step = True

    for p_next in range(start_p + direction * step,
                        end_p + direction * step,
                        direction * step):

        if initial_plateau or not first_step:
            x += dx
            xs.append(x)
            ys.append(current_p)
        xs.append(x)
        ys.append(p_next)
        current_p = p_next
        first_step = False

    return xs, ys, x


def cycles_sequence(start_x,
                    high_levels,
                    cycles_per_level=10,
                    cycle_width=0.18,
                    p_low=0):

    xs, ys = [], []
    x = start_x

    for p_high in high_levels:
        for _ in range(cycles_per_level):

            xs += [
                x,
                x,
                x + cycle_width / 2,
                x + cycle_width / 2,
                x + cycle_width
            ]

            ys += [
                p_low,
                p_high,
                p_high,
                p_low,
                p_low
            ]

            x += cycle_width

    return xs, ys, x


def draw_pressure_bands(ax,
                       start_x,
                       high_levels,
                       cycles_per_level=10,
                       cycle_width=0.18,
                       p_low=0,
                       color="0.9",
                       alpha=0.22,
                       edgecolor="black",
                       edgewidth=1.6):
    if not high_levels or cycles_per_level <= 0 or cycle_width <= 0:
        return start_x

    total_width = len(high_levels) * cycles_per_level * cycle_width
    x = start_x
    verts = [(start_x, p_low)]

    for p_high in high_levels:
        verts.append((x, p_high))
        x += cycles_per_level * cycle_width
        verts.append((x, p_high))

    verts.append((start_x + total_width, p_low))
    band = Polygon(verts, closed=True, facecolor=color, edgecolor=edgecolor,
                   linewidth=edgewidth, alpha=alpha)
    ax.add_patch(band)
    return start_x + total_width


# ============================================================
# CREATE LOADING SEQUENCES
# ============================================================

transition_margin = 0.25

# Progressive loading
pre_x, pre_y, x_pre_end = ramp_sequence(
    start_x=0,
    start_p=0,
    end_p=100,
    step=10,
    dx=0.28
)

step_width = 0.28
x_pre_end_dash = x_pre_end + step_width + transition_margin
x_pre_end_gap_end = x_pre_end_dash + transition_margin

# Hold at 100 kPa for one full step width, then drop to zero and keep the same margin to the dashed separator.
pre_x += [x_pre_end + step_width, x_pre_end + step_width]
pre_y += [100, 0]

_, _, x_prog_end = cycles_sequence(
    start_x=x_pre_end_gap_end,
    high_levels=[50, 100, 150, 200, 250, 300],
    cycles_per_level=10,
    cycle_width=0.075
)

x_post_dash = x_prog_end + transition_margin
x_post_start = x_post_dash + transition_margin
post_x, post_y, x_post_end = ramp_sequence(
    start_x=x_post_start,
    start_p=0,
    end_p=100,
    step=10,
    dx=0.28,
)

# Drop to zero at end using the same ramp step width.
post_x += [post_x[-1] + step_width, post_x[-1] + step_width]
post_y += [100, 0]


# Fatigue loading
pre2_x, pre2_y, x2_pre_end = ramp_sequence(
    start_x=0,
    start_p=0,
    end_p=100,
    step=10,
    dx=0.28
)

x2_pre_end_dash = x2_pre_end + step_width + transition_margin
x2_pre_end_gap_end = x2_pre_end_dash + transition_margin

# Hold at 100 kPa for one full step width, then drop to zero and keep the same margin to the dashed separator.
pre2_x += [x2_pre_end + step_width, x2_pre_end + step_width]
pre2_y += [100, 0]

_, _, x2_fat_end = cycles_sequence(
    start_x=x2_pre_end_gap_end,
    high_levels=[100],
    cycles_per_level=80,   # schematic subset of 500 cycles
    cycle_width=0.06
)

x2_post_dash = x2_fat_end + transition_margin
x2_post_start = x2_post_dash + transition_margin
post2_x, post2_y, x2_post_end = ramp_sequence(
    start_x=x2_post_start,
    start_p=0,
    end_p=100,
    step=10,
    dx=0.28,
)

post2_x += [post2_x[-1] + step_width, post2_x[-1] + step_width]
post2_y += [100, 0]


# ============================================================
# PLOTTING
# ============================================================

configure_style()
fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 6.8)
)

line_width = 2.0
dash_line_width = 1.2


def format_axis(ax,
                title,
                main_label,
                x_pre_end,
                x_post_start,
                ymax=320):

    ax.set_ylim(-15, ymax)

    if ymax > 150:
        ax.set_yticks([0, 100, 200, 300])
    else:
        ax.set_yticks([0, 50, 100])

    ax.set_ylabel(r"Gauge Pressure [kPa]", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    ax.set_title(
        title,
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=10
    )

    # Clean axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Grid
    ax.grid(axis="y", alpha=0.25)

    # Remove x ticks
    ax.set_xticks([])

    # Shaded regions
    ax.axvspan(0, x_pre_end, alpha=0.08)
    ax.axvspan(x_pre_end, x_post_start, alpha=0.04)
    ax.axvspan(x_post_start, ax.get_xlim()[1], alpha=0.08)

    # Vertical separators
    ax.axvline(
        x_pre_end,
        linestyle="--",
        linewidth=dash_line_width
    )

    ax.axvline(
        x_post_start,
        linestyle="--",
        linewidth=dash_line_width
    )

    # Labels
    y_text = ymax * 0.92

    ax.text(
        x_pre_end / 2,
        y_text,
        "Pre-ramp",
        ha="center",
        va="top",
        fontsize=13
    )

    ax.text(
        (x_pre_end + x_post_start) / 2,
        y_text,
        main_label,
        ha="center",
        va="top",
        fontsize=13
    )

    ax.text(
        (x_post_start + ax.get_xlim()[1]) / 2,
        y_text,
        "Post-ramp",
        ha="center",
        va="top",
        fontsize=13
    )


# ============================================================
# TOP PLOT — PROGRESSIVE LOADING
# ============================================================

axes[0].plot(pre_x, pre_y, color="black", linewidth=line_width)
axes[0].plot(post_x, post_y, color="black", linewidth=line_width)

draw_pressure_bands(
    axes[0],
    start_x=x_pre_end_gap_end,
    high_levels=[50, 100, 150, 200, 250, 300],
    cycles_per_level=10,
    cycle_width=0.075,
    p_low=0,
    color="0.8",
    alpha=0.9,
    edgecolor="black",
    edgewidth=1.8,
)

axes[0].set_xlim(-0.2, x_post_end + 0.6)

format_axis(
    axes[0],
    "Progressive pressure loading",
    "Progressive pressure band \n50–300 kPa",
    x_pre_end_dash,
    x_prog_end + transition_margin,
    ymax=330
)

# ============================================================
# BOTTOM PLOT — FATIGUE LOADING
# ============================================================

axes[1].plot(pre2_x, pre2_y, color="black", linewidth=line_width)
axes[1].plot(post2_x, post2_y, color="black", linewidth=line_width)

draw_pressure_bands(
    axes[1],
    start_x=x2_pre_end_gap_end,
    high_levels=[100],
    cycles_per_level=80,
    cycle_width=0.06,
    p_low=0,
    color="0.8",
    alpha=0.9,
    edgecolor="black",
    edgewidth=1.8,
)

axes[1].set_xlim(-0.2, x2_post_end + 0.6)

format_axis(
    axes[1],
    "Cyclic fatigue loading",
    "Fatigue pressure band \n0–100 kPa",
    x2_pre_end_dash,
    x2_fat_end + transition_margin,
    ymax=130
)

axes[1].set_xlabel("Test sequence", fontsize=13)


# ============================================================
# FINAL FORMATTING
# ============================================================

fig.tight_layout(rect=[0, 0, 1, 1])

plt.show()

# Optional save
fig.savefig("pressure_protocols.png", dpi=300, bbox_inches="tight")
# fig.savefig("pressure_protocols.pdf", bbox_inches="tight")