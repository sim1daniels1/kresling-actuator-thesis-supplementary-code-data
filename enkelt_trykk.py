import csv
import time
from datetime import datetime
import serial

# ===================== SETTINGS =====================
PORT = "COM3"          # f.eks. "COM3" på Windows, "/dev/ttyACM0" på Linux
BAUD = 115200
OUTFILE = "pressure_log_application.csv"

# Hvis Arduino sender: "millis,PkPa"
# Eksempel: "12345,101.7"
# ===================================================


def parse_line(line: str):
    """
    Return (arduino_millis, pressure_kPa) or (None, None) if parse fails.
    Accepts:
      - "millis,pressure"
      - "millis,raw,vout,pressure" (tar første og siste)
    """
    parts = [p.strip() for p in line.split(",") if p.strip() != ""]
    if len(parts) < 2:
        return None, None

    try:
        arduino_millis = int(float(parts[0]))
        pressure_kpa = float(parts[-1])
        return arduino_millis, pressure_kpa
    except ValueError:
        return None, None


def main():
    print(f"Opening serial: {PORT} @ {BAUD}")
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2.0)  # gi Arduino tid til reset

    with open(OUTFILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arduino_millis", "pressure_kPa", "raw_line"])

        print(f"Logging to: {OUTFILE}")
        print("Press Ctrl+C to stop.\n")

        try:
            while True:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                pc_iso = datetime.now().isoformat(timespec="milliseconds")
                pc_epoch = time.time()

                arduino_ms, p_kpa = parse_line(line)

                # terminal output
                if arduino_ms is not None:
                    print(f"{pc_iso} | {arduino_ms:>8} ms | {p_kpa:>8.2f} kPa | {line}")
                else:
                    print(f"{pc_iso} | (unparsed) | {line}")

                w.writerow([pc_iso, f"{pc_epoch:.6f}", arduino_ms, p_kpa, line])
                f.flush()

        except KeyboardInterrupt:
            print("\nStopped by user.")

        finally:
            ser.close()
            print("Serial closed.")


if __name__ == "__main__":
    main()