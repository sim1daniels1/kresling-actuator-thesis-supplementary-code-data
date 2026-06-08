import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PIXEL_FILE = "10x_realtest01_yellow.csv"
PRESSURE_FILE = "10x_fillet_test_1.csv"

FPS = 30.0
REAL_DURATION_S = 11 * 60 + 6
VIDEO_DURATION_S = 7 * 60 + 46
TIME_SCALE_INIT = REAL_DURATION_S / VIDEO_DURATION_S

PIXEL_SMOOTH_WINDOW = 21
PRESSURE_SMOOTH_WINDOW = 11

TIME_SCALE_MIN = TIME_SCALE_INIT * 0.97
TIME_SCALE_MAX = TIME_SCALE_INIT * 1.03
TIME_SCALE_STEP = 0.00025
FIT_DT = 0.05
OFFSET_REFINE_RANGE_S = 1.5
OFFSET_REFINE_STEP_S = 0.05

INITIAL_RAMP_STATE = "RAMP_PRE"
FINAL_RAMP_STATE = "RAMP_POST"
W_ANCHOR = 1.0
W_PRE_ANCHOR = 2.0
W_INITIAL = 1.5
W_FINAL = 1.5
W_CYCLIC = 2.5
W_POST_ANCHOR = 1.5
W_CYCLIC_BANDS = 6.0

CYCLIC_PRESSURE_BANDS = [
    (140.0, 220.0),
    (220.0, 320.0),
]

RELEASE_SEARCH_START_S = 54.5
RELEASE_SEARCH_END_S = 57.5
HEIGHT_MID_DIRECTION = "rising"

RAMP_PAD_BEFORE = 3.0
RAMP_PAD_AFTER = 3.0
ANCHOR_PAD_BEFORE = 4.0
ANCHOR_PAD_AFTER = 4.0

INITIAL_EXPORT_FILE = "aligned_initial_ramp.csv"
FINAL_EXPORT_FILE = "aligned_final_ramp.csv"
FULL_EXPORT_FILE = "10x_aligned_full_test.csv"
SEARCH_RESULTS_FILE = "alignment_scale_search_results.csv"
ZERO_KPA_MAX = 2.0
REF_FALLBACK_N = 20


def rolling_mean(series, window):
    return series.rolling(window=window, center=True, min_periods=1).mean()


def crossing_time(t, y, level, direction="falling"):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    for i in range(len(y) - 1):
        y0, y1 = y[i], y[i + 1]
        if direction == "falling":
            cond = (y0 >= level) and (y1 <= level)
        else:
            cond = (y0 <= level) and (y1 >= level)
        if cond and (y1 != y0):
            frac = (level - y0) / (y1 - y0)
            return t[i] + frac * (t[i + 1] - t[i])
    return np.nan


def load_pressure_csv(path):
    pr = pd.read_csv(path)
    for col in ["timestamp_ms", "P_g", "rampTarget_g"]:
        pr[col] = pd.to_numeric(pr[col], errors="coerce")
    if "cycles" in pr.columns:
        pr["cycles"] = pd.to_numeric(pr["cycles"], errors="coerce")
    if "cyclesTgt" in pr.columns:
        pr["cyclesTgt"] = pd.to_numeric(pr["cyclesTgt"], errors="coerce")
    pr["state"] = pr["state"].astype(str)
    pr = pr.dropna(subset=["timestamp_ms", "P_g", "rampTarget_g"]).reset_index(drop=True)
    pr["t_s"] = pr["timestamp_ms"] / 1000.0
    pr["P_g_smooth"] = rolling_mean(pr["P_g"], PRESSURE_SMOOTH_WINDOW)
    return pr


def normalize_01(x):
    x = np.asarray(x, dtype=float)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or np.isclose(xmax - xmin, 0):
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def load_pixel_csv(path):
    px = pd.read_csv(path)
    px["Frame"] = pd.to_numeric(px["Frame"], errors="coerce")
    px["Height_yellow_px"] = pd.to_numeric(px["Height_yellow_px"], errors="coerce")
    return px.dropna(subset=["Frame", "Height_yellow_px"]).reset_index(drop=True)


def build_pixel_with_scale(px_raw, time_scale, smooth_window):
    px = px_raw.copy()
    px["t_real_s"] = px["Frame"] / FPS * time_scale
    px["height_smooth"] = rolling_mean(px["Height_yellow_px"], smooth_window)
    return px


def first_time_of_state(pr, state_name):
    rows = pr[pr["state"] == state_name]
    if len(rows) == 0:
        raise ValueError(f"No rows found for state '{state_name}'")
    return rows["t_s"].iloc[0]


def last_time_of_state(pr, state_name):
    rows = pr[pr["state"] == state_name]
    if len(rows) == 0:
        raise ValueError(f"No rows found for state '{state_name}'")
    return rows["t_s"].iloc[-1]


def first_cycle_time_after_pre(pr):
    pre_end = last_time_of_state(pr, INITIAL_RAMP_STATE)
    after_pre = pr[pr["t_s"] > pre_end].copy()
    if "cycles" in after_pre.columns:
        cyc = pd.to_numeric(after_pre["cycles"], errors="coerce")
        hit = after_pre[cyc > 0]
        if len(hit) > 0:
            return hit["t_s"].iloc[0]
    hit = after_pre[~after_pre["state"].isin([INITIAL_RAMP_STATE])]
    if len(hit) > 0:
        return hit["t_s"].iloc[0]
    return pre_end


def detect_release_midpoints(pr, px, t0, t1):
    pr_rel = pr[(pr["t_s"] >= t0) & (pr["t_s"] <= t1)].copy()
    px_rel = px[(px["t_real_s"] >= t0) & (px["t_real_s"] <= t1)].copy()
    if len(pr_rel) < 5 or len(px_rel) < 5:
        return np.nan, np.nan

    p_high = np.percentile(pr_rel["P_g_smooth"], 90)
    p_low = np.percentile(pr_rel["P_g_smooth"], 10)
    p_mid = 0.5 * (p_high + p_low)
    t_pressure_mid = crossing_time(
        pr_rel["t_s"].values,
        pr_rel["P_g_smooth"].values,
        p_mid,
        direction="falling",
    )

    h_low = np.percentile(px_rel["height_smooth"], 10)
    h_high = np.percentile(px_rel["height_smooth"], 90)
    h_mid = 0.5 * (h_low + h_high)
    t_height_mid = crossing_time(
        px_rel["t_real_s"].values,
        px_rel["height_smooth"].values,
        h_mid,
        direction=HEIGHT_MID_DIRECTION,
    )
    return t_pressure_mid, t_height_mid


def detect_state_midpoints(pr, px, state_name, pressure_direction, height_direction):
    pr_state = pr[pr["state"] == state_name].copy().reset_index(drop=True)
    if len(pr_state) < 5:
        return np.nan, np.nan

    t0 = pr_state["t_s"].iloc[0]
    t1 = pr_state["t_s"].iloc[-1]
    px_state = px[(px["t_real_s"] >= t0) & (px["t_real_s"] <= t1)].copy()
    if len(px_state) < 5:
        return np.nan, np.nan

    p_high = np.percentile(pr_state["P_g_smooth"], 90)
    p_low = np.percentile(pr_state["P_g_smooth"], 10)
    p_mid = 0.5 * (p_high + p_low)
    t_pressure_mid = crossing_time(
        pr_state["t_s"].values,
        pr_state["P_g_smooth"].values,
        p_mid,
        direction=pressure_direction,
    )

    h_low = np.percentile(px_state["height_smooth"], 10)
    h_high = np.percentile(px_state["height_smooth"], 90)
    h_mid = 0.5 * (h_low + h_high)
    t_height_mid = crossing_time(
        px_state["t_real_s"].values,
        px_state["height_smooth"].values,
        h_mid,
        direction=height_direction,
    )
    return t_pressure_mid, t_height_mid


def ramp_shape_error(pr, px_aligned, state_name, dt):
    pr_state = pr[pr["state"] == state_name].copy().reset_index(drop=True)
    if len(pr_state) < 10:
        return np.nan

    t0 = pr_state["t_s"].iloc[0]
    t1 = pr_state["t_s"].iloc[-1]
    t_grid = np.arange(t0, t1 + dt, dt)

    p = np.interp(t_grid, pr_state["t_s"], pr_state["P_g_smooth"])
    h = np.interp(t_grid, px_aligned["t_aligned_s"], px_aligned["height_smooth"], left=np.nan, right=np.nan)

    valid = ~np.isnan(h)
    if valid.sum() < 20:
        return np.nan

    p_n = normalize_01(p[valid])
    h_inv_n = normalize_01(np.nanmax(h[valid]) - h[valid])
    return np.mean((h_inv_n - p_n) ** 2)


def cyclic_section_error(pr, px_aligned, t0, t1, dt):
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.nan

    pr_seg = pr[(pr["t_s"] >= t0) & (pr["t_s"] <= t1)].copy().reset_index(drop=True)
    if len(pr_seg) < 20:
        return np.nan

    t_grid = np.arange(t0, t1 + dt, dt)
    p = np.interp(t_grid, pr_seg["t_s"], pr_seg["P_g_smooth"])
    h = np.interp(t_grid, px_aligned["t_aligned_s"], px_aligned["height_smooth"], left=np.nan, right=np.nan)

    valid = ~np.isnan(h)
    if valid.sum() < 50:
        return np.nan

    p_n = normalize_01(p[valid])
    h_inv_n = normalize_01(np.nanmax(h[valid]) - h[valid])
    return np.mean((h_inv_n - p_n) ** 2)


def pressure_band_error(pr, px_aligned, t0, t1, p_min, p_max, dt):
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.nan

    t_grid = np.arange(t0, t1 + dt, dt)
    if len(t_grid) < 20:
        return np.nan

    p = np.interp(t_grid, pr["t_s"], pr["P_g_smooth"], left=np.nan, right=np.nan)
    h = np.interp(t_grid, px_aligned["t_aligned_s"], px_aligned["height_smooth"], left=np.nan, right=np.nan)

    valid = np.isfinite(p) & np.isfinite(h) & (p >= p_min) & (p <= p_max)
    if valid.sum() < 40:
        return np.nan

    p_n = normalize_01(p[valid])
    h_inv_n = normalize_01(np.nanmax(h[valid]) - h[valid])
    return np.mean((h_inv_n - p_n) ** 2)


def trim_to_overlap(pr, px):
    t0 = max(pr["t_s"].min(), px["t_aligned_s"].min())
    t1 = min(pr["t_s"].max(), px["t_aligned_s"].max())
    if t1 <= t0:
        raise ValueError("No overlapping time range after scaling/alignment.")
    pr_trim = pr[(pr["t_s"] >= t0) & (pr["t_s"] <= t1)].copy().reset_index(drop=True)
    px_trim = px[(px["t_aligned_s"] >= t0) & (px["t_aligned_s"] <= t1)].copy().reset_index(drop=True)
    return pr_trim, px_trim, t0, t1


def compute_reference_height(pr_ref, px_aligned, zero_kpa_max, fallback_n):
    h_interp = np.interp(pr_ref["t_s"], px_aligned["t_aligned_s"], px_aligned["height_smooth"], left=np.nan, right=np.nan)
    zero_mask = (pr_ref["P_g_smooth"] <= zero_kpa_max) & np.isfinite(h_interp)
    if zero_mask.sum() >= 3:
        return float(np.nanmean(h_interp[zero_mask]))
    finite_h = h_interp[np.isfinite(h_interp)]
    if len(finite_h) == 0:
        raise ValueError("Could not compute reference height from aligned data.")
    return float(np.nanmean(finite_h[:fallback_n]))


def export_window(pr, px, t0, t1, out_file, h_ref):
    pr_seg = pr[(pr["t_s"] >= t0) & (pr["t_s"] <= t1)].copy().reset_index(drop=True)
    out = pd.DataFrame({
        "time_s": pr_seg["t_s"],
        "state": pr_seg["state"],
        "P_g": pr_seg["P_g"],
        "P_g_smooth": pr_seg["P_g_smooth"],
        "rampTarget_g": pr_seg["rampTarget_g"],
        "rampHold": pr_seg["rampHold"] if "rampHold" in pr_seg.columns else np.nan,
    })
    out["Height_yellow_px"] = np.interp(out["time_s"], px["t_aligned_s"], px["Height_yellow_px"], left=np.nan, right=np.nan)
    out["Height_yellow_px_smooth"] = np.interp(out["time_s"], px["t_aligned_s"], px["height_smooth"], left=np.nan, right=np.nan)
    out["compression_pct"] = (h_ref - out["Height_yellow_px"]) / h_ref * 100.0
    out["compression_pct_smooth"] = (h_ref - out["Height_yellow_px_smooth"]) / h_ref * 100.0
    out.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")


def plot_window(ax, pr, px, t0, t1, title):
    pr_plot = pr[(pr["t_s"] >= t0) & (pr["t_s"] <= t1)].copy()
    px_plot = px[(px["t_aligned_s"] >= t0) & (px["t_aligned_s"] <= t1)].copy()
    ax.plot(px_plot["t_aligned_s"], px_plot["Height_yellow_px"], alpha=0.35, linewidth=1.0, label="Height raw")
    ax.plot(px_plot["t_aligned_s"], px_plot["height_smooth"], linewidth=2.0, label="Height smooth")
    ax.set_ylabel("Height [px]")
    ax.set_title(title)
    ax.grid(True)
    ax_r = ax.twinx()
    ax_r.plot(pr_plot["t_s"], pr_plot["P_g"], "--", alpha=0.35, linewidth=1.0, label="Pressure raw")
    ax_r.plot(pr_plot["t_s"], pr_plot["P_g_smooth"], "--", linewidth=2.0, label="Pressure smooth")
    ax_r.plot(pr_plot["t_s"], pr_plot["rampTarget_g"], ":", linewidth=1.5, label="Ramp target")
    ax_r.set_ylabel("Pressure [kPa]")
    return ax, ax_r


pr = load_pressure_csv(PRESSURE_FILE)
px_raw = load_pixel_csv(PIXEL_FILE)

t_cycles_start = first_cycle_time_after_pre(pr)
t_post_start = first_time_of_state(pr, FINAL_RAMP_STATE)
results = []
candidate_offsets = np.arange(-OFFSET_REFINE_RANGE_S, OFFSET_REFINE_RANGE_S + OFFSET_REFINE_STEP_S, OFFSET_REFINE_STEP_S)

for scale in np.arange(TIME_SCALE_MIN, TIME_SCALE_MAX + TIME_SCALE_STEP, TIME_SCALE_STEP):
    px_candidate = build_pixel_with_scale(px_raw, scale, PIXEL_SMOOTH_WINDOW)
    t_pressure_mid, t_height_mid = detect_release_midpoints(
        pr, px_candidate, RELEASE_SEARCH_START_S, RELEASE_SEARCH_END_S
    )
    if not np.isfinite(t_pressure_mid) or not np.isfinite(t_height_mid):
        continue

    t_pre_pressure_mid, t_pre_height_mid = detect_state_midpoints(
        pr, px_candidate, INITIAL_RAMP_STATE, pressure_direction="rising", height_direction="falling"
    )
    t_post_pressure_mid, t_post_height_mid = detect_state_midpoints(
        pr, px_candidate, FINAL_RAMP_STATE, pressure_direction="rising", height_direction="falling"
    )
    release_offset = t_pressure_mid - t_height_mid

    for delta_offset in candidate_offsets:
        offset = release_offset + delta_offset
        px_candidate["t_aligned_s"] = px_candidate["t_real_s"] + offset

        anchor_err = delta_offset ** 2
        pre_anchor_err = np.nan
        if np.isfinite(t_pre_pressure_mid) and np.isfinite(t_pre_height_mid):
            pre_anchor_err = (t_pre_pressure_mid - (t_pre_height_mid + offset)) ** 2
        post_anchor_err = np.nan
        if np.isfinite(t_post_pressure_mid) and np.isfinite(t_post_height_mid):
            post_anchor_err = (t_post_pressure_mid - (t_post_height_mid + offset)) ** 2
        initial_err = ramp_shape_error(pr, px_candidate, INITIAL_RAMP_STATE, FIT_DT)
        final_err = ramp_shape_error(pr, px_candidate, FINAL_RAMP_STATE, FIT_DT)
        cyclic_err = cyclic_section_error(pr, px_candidate, t_cycles_start, t_post_start, FIT_DT)
        band_errors = [
            pressure_band_error(pr, px_candidate, t_cycles_start, t_post_start, p_min, p_max, FIT_DT)
            for p_min, p_max in CYCLIC_PRESSURE_BANDS
        ]
        finite_band_errors = [err for err in band_errors if np.isfinite(err)]
        if (
            not np.isfinite(initial_err)
            or not np.isfinite(final_err)
            or not np.isfinite(cyclic_err)
            or not finite_band_errors
        ):
            continue

        total_err = (
            W_ANCHOR * anchor_err
            + W_INITIAL * initial_err
            + W_FINAL * final_err
            + W_CYCLIC * cyclic_err
            + W_CYCLIC_BANDS * float(np.mean(finite_band_errors))
        )
        if np.isfinite(pre_anchor_err):
            total_err += W_PRE_ANCHOR * pre_anchor_err
        if np.isfinite(post_anchor_err):
            total_err += W_POST_ANCHOR * post_anchor_err

        results.append({
            "scale": scale,
            "release_offset": release_offset,
            "delta_offset": delta_offset,
            "offset": offset,
            "t_pre_pressure_mid": t_pre_pressure_mid,
            "t_pre_height_mid": t_pre_height_mid,
            "t_pressure_mid": t_pressure_mid,
            "t_height_mid": t_height_mid,
            "t_post_pressure_mid": t_post_pressure_mid,
            "t_post_height_mid": t_post_height_mid,
            "anchor_err": anchor_err,
            "pre_anchor_err": pre_anchor_err,
            "post_anchor_err": post_anchor_err,
            "initial_err": initial_err,
            "final_err": final_err,
            "cyclic_err": cyclic_err,
            "cyclic_band_err_mean": float(np.mean(finite_band_errors)),
            "cyclic_band_err_140_220": band_errors[0],
            "cyclic_band_err_220_320": band_errors[1],
            "total_err": total_err,
        })

if not results:
    raise ValueError("Optimization failed. No valid scale candidates found.")

res_df = pd.DataFrame(results)
best = res_df.loc[res_df["total_err"].idxmin()]
BEST_SCALE = float(best["scale"])
BEST_OFFSET = float(best["offset"])
BEST_RELEASE_OFFSET = float(best["release_offset"])
BEST_DELTA_OFFSET = float(best["delta_offset"])
T_PRE_PRESSURE_MID = float(best["t_pre_pressure_mid"])
T_PRE_HEIGHT_MID = float(best["t_pre_height_mid"])
T_PRESSURE_MID = float(best["t_pressure_mid"])
T_HEIGHT_MID = float(best["t_height_mid"])
T_POST_PRESSURE_MID = float(best["t_post_pressure_mid"])
T_POST_HEIGHT_MID = float(best["t_post_height_mid"])

px = build_pixel_with_scale(px_raw, BEST_SCALE, PIXEL_SMOOTH_WINDOW)
px["t_aligned_s"] = px["t_real_s"] + BEST_OFFSET
pr, px, T_TRIM_START, T_TRIM_END = trim_to_overlap(pr, px)
pr_ref = pr[pr["t_s"] <= t_cycles_start].copy()
h_ref = compute_reference_height(pr_ref, px, ZERO_KPA_MAX, REF_FALLBACK_N)

initial_t0 = max(first_time_of_state(pr, INITIAL_RAMP_STATE), T_TRIM_START)
initial_t1 = min(t_cycles_start, T_TRIM_END)
final_t0 = max(first_time_of_state(pr, FINAL_RAMP_STATE), T_TRIM_START)
final_t1 = T_TRIM_END

export_window(pr, px, initial_t0, initial_t1, INITIAL_EXPORT_FILE, h_ref)
export_window(pr, px, final_t0, final_t1, FINAL_EXPORT_FILE, h_ref)
export_window(pr, px, T_TRIM_START, T_TRIM_END, FULL_EXPORT_FILE, h_ref)
res_df.to_csv(SEARCH_RESULTS_FILE, index=False)

print(f"Initial TIME_SCALE estimate : {TIME_SCALE_INIT:.6f}")
print(f"Best TIME_SCALE             : {BEST_SCALE:.6f}")
print(f"Pre-ramp pressure midpoint  : {T_PRE_PRESSURE_MID:.3f} s")
print(f"Pre-ramp height midpoint    : {T_PRE_HEIGHT_MID:.3f} s")
print(f"Release anchor offset       : {BEST_RELEASE_OFFSET:.3f} s")
print(f"Offset refinement           : {BEST_DELTA_OFFSET:.3f} s")
print(f"Pressure release midpoint   : {T_PRESSURE_MID:.3f} s")
print(f"Height recovery midpoint    : {T_HEIGHT_MID:.3f} s")
print(f"Applied offset              : {BEST_OFFSET:.3f} s")
print(f"Post-ramp pressure midpoint : {T_POST_PRESSURE_MID:.3f} s")
print(f"Post-ramp height midpoint   : {T_POST_HEIGHT_MID:.3f} s")
print(f"Trimmed overlap             : {T_TRIM_START:.3f} to {T_TRIM_END:.3f} s")
print(f"Reference height:           {h_ref:.3f} px")
print(f"Reference window end:       {t_cycles_start:.3f} s")
print(f"Candidates checked:         {len(res_df)}")
print(f"Best total error:           {best['total_err']:.6f}")
print(f"Mean cyclic band error:     {best['cyclic_band_err_mean']:.6f}")
print(f"Saved: {SEARCH_RESULTS_FILE}")

pr_pre = pr[pr["state"] == INITIAL_RAMP_STATE].copy().reset_index(drop=True)
pr_post = pr[pr["state"] == FINAL_RAMP_STATE].copy().reset_index(drop=True)
fig, axes = plt.subplots(3, 1, figsize=(12, 13))
ax1, ax1r = plot_window(axes[0], pr, px, pr_pre["t_s"].iloc[0] - RAMP_PAD_BEFORE, pr_pre["t_s"].iloc[-1] + RAMP_PAD_AFTER, "Initial ramp check")
ax2, ax2r = plot_window(axes[1], pr, px, T_PRESSURE_MID - ANCHOR_PAD_BEFORE, T_PRESSURE_MID + ANCHOR_PAD_AFTER, "Drop-anchor midpoint check")
axes[1].axvline(T_PRESSURE_MID, color="k", linestyle=":", linewidth=1.2, label="Pressure midpoint")
axes[1].axvline(T_HEIGHT_MID + BEST_OFFSET, color="gray", linestyle="--", linewidth=1.2, label="Height midpoint")
ax3, ax3r = plot_window(axes[2], pr, px, pr_post["t_s"].iloc[0] - RAMP_PAD_BEFORE, pr["t_s"].iloc[-1], "Final ramp check")
axes[2].set_xlabel("Time [s]")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1r.get_legend_handles_labels()
axes[0].legend(lines1 + lines2, labels1 + labels2, loc="best")
plt.tight_layout()
plt.show()

fig_full, ax_full = plt.subplots(figsize=(12, 5))
compression_smooth = (h_ref - np.interp(pr["t_s"], px["t_aligned_s"], px["height_smooth"], left=np.nan, right=np.nan)) / h_ref * 100.0
ax_full.plot(pr["t_s"], compression_smooth, linewidth=2.0, label="Compression smooth")
ax_full.set_xlabel("Time [s]")
ax_full.set_ylabel("Compression [%]")
ax_full.grid(True)
ax_full_r = ax_full.twinx()
ax_full_r.plot(pr["t_s"], pr["P_g_smooth"], "--", linewidth=2.0, label="Pressure smooth")
ax_full_r.plot(pr["t_s"], pr["rampTarget_g"], ":", linewidth=1.3, label="Ramp target")
ax_full_r.set_ylabel("Pressure [kPa]")
lines1, labels1 = ax_full.get_legend_handles_labels()
lines2, labels2 = ax_full_r.get_legend_handles_labels()
ax_full.legend(lines1 + lines2, labels1 + labels2, loc="best")
ax_full.set_title("Aligned full ramp-up test")
plt.tight_layout()
plt.show()
