# log_pressure_ramp_cycle_sanity.py
import serial
import csv
import time

PORT = "COM3"
BAUD = 115200
OUTFILE = "euromex_test.csv"

START_DELAY_SEC = 10

FIELDS_TO_LOG = [
    "P", "rampTarget", "rampHold", "P_low", "P_high",
    "posSteps", "state", "cycles", "cyclesTgt", "running"
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

with serial.Serial(PORT, BAUD, timeout=1) as ser, open(OUTFILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp_ms"] + FIELDS_TO_LOG)

    print(f"Opened {PORT} at {BAUD} baud")
    time.sleep(2.0)
    ser.reset_input_buffer()

    print("Logging started BEFORE starting test...")

    start_ms = int(time.time() * 1000)
    start_command_sent = False
    start_send_time = start_ms + int(START_DELAY_SEC * 1000)

    last_state = None
    last_target = None
    last_cycles = None

    try:
        while True:
            now_ms = int(time.time() * 1000)

            if not start_command_sent and now_ms >= start_send_time:
                ser.write(b's\n')
                ser.flush()
                start_command_sent = True
                print(">>> Sent START command to Arduino")

            raw = ser.readline().decode(errors="replace").strip()
            if not raw:
                continue

            # Ignore non key-value info lines
            if "P=" not in raw:
                print("INFO:", raw)
                continue

            d = parse_kv_line(raw)

            # Sanity: require pressure
            p = to_float(d.get("P"))
            if p is None:
                continue

            # Build row
            row = [now_ms - start_ms]
            for k in FIELDS_TO_LOG:
                row.append(d.get(k, ""))

            writer.writerow(row)
            f.flush()

            # Pretty event prints
            state = d.get("state")
            target = to_float(d.get("rampTarget")) if d.get("rampTarget") is not None else None
            cycles = to_int(d.get("cycles"))

            if state != last_state:
                print(f"[EVENT] state: {last_state} -> {state}   (P={p:.2f})")
                last_state = state

            if target is not None and target != last_target:
                print(f"[EVENT] rampTarget -> {target:.1f} kPa   (P={p:.2f})")
                last_target = target

            if cycles is not None and cycles != last_cycles:
                print(f"[EVENT] cycles -> {cycles}   (P={p:.2f}, state={state})")
                last_cycles = cycles

            # Optional: lightweight live print
            # print(f"{row[0]} ms  P={p:.2f}  target={d.get('rampTarget')}  state={state}")

    except KeyboardInterrupt:
        print("\nStopping...")
        try:
            ser.write(b'x\n')
            ser.flush()
            print(">>> Sent STOP command to Arduino")
            time.sleep(0.2)
        except Exception as e:
            print("Error while sending stop:", e)

print("Done.")