# log_pressure_step_fracture_abs.py
import serial
import csv
import time

PORT = "COM3"
BAUD = 115200

OUTFILE = "5x_v1_frac_10cycles.csv"
START_DELAY_SEC = 5

# Matches the CURRENT Arduino telemetry keys:
# P (abs estimate), P_g (ABP gauge), Pabs_mpx (MPX abs raw), Patm_hat (estimated ambient abs),
# P_init (abs at calibration), Pmin, Phigh, rampTarget, rampHold, cyclesStep, posSteps, DIR_SIGN, state, running
FIELDS_TO_LOG = [
    "P", "P_g", "Pabs_mpx", "Patm_hat", "P_init",
    "Pmin", "Phigh",
    "rampTarget", "rampHold",
    "cyclesStep", "posSteps", "DIR_SIGN",
    "state", "running"
]

def parse_kv_line(raw: str) -> dict:
    d = {}
    parts = raw.split(",")
    for p in parts:
        p = p.strip()
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        d[k.strip()] = v.strip()
    return d

def to_float(x):
    try:
        return float(x)
    except:
        return None

def to_int(x):
    try:
        return int(float(x))
    except:
        return None

def extract_cal_complete(raw: str):
    """
    Arduino line looks like:
      CAL complete. P_init_abs=100.12 kPa, Patm_hat=98.34 kPa. -> RAMP_PRE
    Returns (P_init_abs, Patm_hat) if found, else (None, None)
    """
    if "CAL complete" not in raw:
        return None, None

    cleaned = (
        raw.replace("=", " ")
           .replace(",", " ")
           .replace("kPa", " ")
           .replace("->", " ")
    )
    toks = cleaned.split()

    p_init_abs = None
    patm_hat = None

    for i, t in enumerate(toks):
        # tokens will include "P_init_abs" and "Patm_hat"
        if t.strip() == "P_init_abs" and i + 1 < len(toks):
            p_init_abs = to_float(toks[i + 1])
        if t.strip() == "Patm_hat" and i + 1 < len(toks):
            patm_hat = to_float(toks[i + 1])

    return p_init_abs, patm_hat


with serial.Serial(PORT, BAUD, timeout=1) as ser, open(OUTFILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp_ms"] + FIELDS_TO_LOG)

    print(f"Opened {PORT} at {BAUD} baud")
    time.sleep(2.0)

    # Important: clear buffered lines so you don't log stale data
    ser.reset_input_buffer()

    print("Logging started BEFORE starting test...")

    start_ms = int(time.time() * 1000)
    start_command_sent = False
    start_send_time = start_ms + int(START_DELAY_SEC * 1000)

    last_state = None
    last_phigh = None
    last_cycles_step = None

    baseline_printed = False

    try:
        while True:
            now_ms = int(time.time() * 1000)

            # Auto-start after delay
            if not start_command_sent and now_ms >= start_send_time:
                ser.write(b's\n')
                ser.flush()
                start_command_sent = True
                print(">>> Sent START command to Arduino")

            raw = ser.readline().decode(errors="replace").strip()
            if not raw:
                continue

            # Print calibration baseline once (do not log as telemetry row)
            if (not baseline_printed) and ("CAL complete" in raw):
                p_init_abs, patm_hat = extract_cal_complete(raw)
                if p_init_abs is not None or patm_hat is not None:
                    if p_init_abs is not None:
                        print(f"[CAL] P_init_abs = {p_init_abs:.2f} kPa")
                    if patm_hat is not None:
                        print(f"[CAL] Patm_hat  = {patm_hat:.2f} kPa")
                    baseline_printed = True
                    continue
                else:
                    print("INFO:", raw)
                    continue

            # Only log telemetry lines that contain P=
            if "P=" not in raw:
                print("INFO:", raw)
                continue

            d = parse_kv_line(raw)

            # Require absolute estimate to be present
            p_abs = to_float(d.get("P"))
            if p_abs is None:
                continue

            # Keep gauge (ABP) for plotting
            p_g = to_float(d.get("P_g"))

            # Build row
            row = [now_ms - start_ms]
            for k in FIELDS_TO_LOG:
                row.append(d.get(k, ""))

            writer.writerow(row)
            f.flush()

            # Event prints
            state = d.get("state")
            phigh = to_float(d.get("Phigh"))
            cycles_step = to_int(d.get("cyclesStep"))

            if state != last_state:
                print(f"[EVENT] state: {last_state} -> {state}   (P_abs={p_abs:.2f}, P_g={p_g if p_g is not None else 'NA'})")
                last_state = state

            if phigh is not None and phigh != last_phigh:
                print(f"[EVENT] Phigh_abs -> {phigh:.1f} kPa   (P_abs={p_abs:.2f})")
                last_phigh = phigh
                last_cycles_step = None

            if cycles_step is not None and cycles_step != last_cycles_step:
                print(f"[EVENT] cyclesStep -> {cycles_step}   (P_abs={p_abs:.2f}, state={state}, Phigh={d.get('Phigh')})")
                last_cycles_step = cycles_step

    except KeyboardInterrupt:
        print("\nStopping...")
        try:
            ser.write(b'x\n')
            ser.flush()
            print(">>> Sent STOP command to Arduino")
            time.sleep(0.2)
        except Exception as e:
            print("Error while sending stop:", e)

print(f"Done. CSV written to: {OUTFILE}")