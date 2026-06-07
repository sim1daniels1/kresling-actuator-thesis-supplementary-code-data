import serial
import time

PORT = "COM3"      # change if needed
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # allow Arduino reset

print("Monitoring pressure...\n")

def parse_line(line):
    line = line.strip()
    if not line:
        return None

    # Works for space or comma separated
    if "," in line:
        parts = line.split(",")
    else:
        parts = line.split()

    if len(parts) < 3:
        return None

    try:
        mpx = float(parts[0])
        abp = float(parts[1])
        delta = float(parts[2])
        return mpx, abp, delta
    except:
        return None

try:
    while True:
        raw = ser.readline().decode(errors="ignore")
        data = parse_line(raw)
        if data is None:
            continue

        mpx, abp, delta = data

        print(f"MPX: {mpx:8.2f} kPa | "
              f"ABP: {abp:8.2f} kPa | "
              f"Δ: {delta:7.2f} kPa")

except KeyboardInterrupt:
    print("\nStopped.")
    ser.close()