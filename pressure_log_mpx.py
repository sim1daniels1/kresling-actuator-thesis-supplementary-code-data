import serial
import threading
import time
import csv
import tkinter as tk
from tkinter import ttk
from datetime import datetime


PORT = "COM3"       # change this
BAUD = 115200

CSV_FILE = f"manual_pressure_mpx4250_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


class PressureController:
    def __init__(self, root):
        self.root = root
        self.root.title("Manual Pressure Logger - MPX4250AP")

        self.ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)

        self.running = True

        self.latest = {
            "timestamp_ms": "",
            "P_abs": "",
            "P_g": "",
            "P_atm": "",
        }

        self.csv_file = open(CSV_FILE, "w", newline="")
        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            "pc_time_s",
            "timestamp_ms",
            "P_abs",
            "P_g",
            "P_atm",
            "raw_line",
        ])

        self.build_gui()

        self.thread = threading.Thread(target=self.read_serial, daemon=True)
        self.thread.start()

        self.update_gui()

    def build_gui(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.grid()

        self.pressure_label = ttk.Label(frame, text="P_g: -- kPa", font=("Arial", 24))
        self.pressure_label.grid(row=0, column=0, pady=10)

        self.abs_label = ttk.Label(frame, text="P_abs: -- kPa", font=("Arial", 13))
        self.abs_label.grid(row=1, column=0, pady=4)

        self.atm_label = ttk.Label(frame, text="P_atm: -- kPa", font=("Arial", 13))
        self.atm_label.grid(row=2, column=0, pady=4)

        self.file_label = ttk.Label(frame, text=f"Logging to: {CSV_FILE}")
        self.file_label.grid(row=3, column=0, pady=10)

    def read_serial(self):
        while self.running:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()

                if not line or line.startswith("timestamp"):
                    continue

                parts = line.split(",")

                if len(parts) < 4:
                    continue

                (
                    timestamp_ms,
                    P_abs,
                    P_g,
                    P_atm,
                ) = parts[:4]

                self.latest = {
                    "timestamp_ms": timestamp_ms,
                    "P_abs": P_abs,
                    "P_g": P_g,
                    "P_atm": P_atm,
                }

                self.writer.writerow([
                    time.time(),
                    timestamp_ms,
                    P_abs,
                    P_g,
                    P_atm,
                    line,
                ])
                self.csv_file.flush()

            except Exception as e:
                print("Serial read error:", e)

    def update_gui(self):
        try:
            self.pressure_label.config(text=f"P_g: {float(self.latest['P_g']):.1f} kPa")
            self.abs_label.config(text=f"P_abs: {float(self.latest['P_abs']):.1f} kPa")
            self.atm_label.config(text=f"P_atm: {float(self.latest['P_atm']):.1f} kPa")
        except:
            pass

        if self.running:
            self.root.after(100, self.update_gui)

    def close(self):
        self.running = False
        try:
            self.ser.close()
        except:
            pass

        self.csv_file.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PressureController(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
