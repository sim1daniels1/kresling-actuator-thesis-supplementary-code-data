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
        self.root.title("Manual Pressure Controller - MPX4250AP")

        self.ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)

        self.running = True

        self.latest = {
            "timestamp_ms": "",
            "P_abs": "",
            "P_g": "",
            "P_atm": "",
            "target_gauge_kPa": "",
            "state": "",
            "posSteps": "",
            "DIR_SIGN": "",
        }

        self.csv_file = open(CSV_FILE, "w", newline="")
        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            "pc_time_s",
            "timestamp_ms",
            "P_abs",
            "P_g",
            "P_atm",
            "target_gauge_kPa",
            "state",
            "posSteps",
            "DIR_SIGN"
        ])

        self.build_gui()

        self.thread = threading.Thread(target=self.read_serial, daemon=True)
        self.thread.start()

        self.update_gui()

    def build_gui(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.grid()

        self.pressure_label = ttk.Label(frame, text="P_g: -- kPa", font=("Arial", 20))
        self.pressure_label.grid(row=0, column=0, columnspan=2, pady=10)

        self.abs_label = ttk.Label(frame, text="P_abs: -- kPa", font=("Arial", 13))
        self.abs_label.grid(row=1, column=0, columnspan=2, pady=4)

        self.atm_label = ttk.Label(frame, text="P_atm: -- kPa", font=("Arial", 13))
        self.atm_label.grid(row=2, column=0, columnspan=2, pady=4)

        self.target_label = ttk.Label(frame, text="Target gauge: -- kPa", font=("Arial", 16))
        self.target_label.grid(row=3, column=0, columnspan=2, pady=5)

        self.state_label = ttk.Label(frame, text="State: --", font=("Arial", 14))
        self.state_label.grid(row=4, column=0, columnspan=2, pady=5)

        up_btn = ttk.Button(frame, text="UP +50 kPa", command=self.pressure_up)
        up_btn.grid(row=5, column=0, padx=10, pady=15, ipadx=20, ipady=10)

        down_btn = ttk.Button(frame, text="DOWN -50 kPa", command=self.pressure_down)
        down_btn.grid(row=5, column=1, padx=10, pady=15, ipadx=20, ipady=10)

        zero_btn = ttk.Button(frame, text="ZERO", command=self.zero_pressure)
        zero_btn.grid(row=6, column=0, padx=10, pady=10, ipadx=20, ipady=8)

        stop_btn = ttk.Button(frame, text="STOP", command=self.stop_motor)
        stop_btn.grid(row=6, column=1, padx=10, pady=10, ipadx=20, ipady=8)

        rezero_btn = ttk.Button(frame, text="Re-zero atmospheric", command=self.rezero)
        rezero_btn.grid(row=7, column=0, padx=10, pady=10, ipadx=20, ipady=8)

        invert_btn = ttk.Button(frame, text="Invert direction", command=self.invert_direction)
        invert_btn.grid(row=7, column=1, padx=10, pady=10, ipadx=20, ipady=8)

        self.file_label = ttk.Label(frame, text=f"Logging to: {CSV_FILE}")
        self.file_label.grid(row=8, column=0, columnspan=2, pady=10)

    def send(self, command):
        self.ser.write((command + "\n").encode())

    def pressure_up(self):
        self.send("UP")

    def pressure_down(self):
        self.send("DOWN")

    def zero_pressure(self):
        self.send("ZERO")

    def stop_motor(self):
        self.send("STOP")

    def rezero(self):
        self.send("REZERO")

    def invert_direction(self):
        self.send("I")

    def read_serial(self):
        while self.running:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()

                if not line or line.startswith("timestamp"):
                    continue

                parts = line.split(",")

                if len(parts) != 8:
                    continue

                (
                    timestamp_ms,
                    P_abs,
                    P_g,
                    P_atm,
                    target_gauge_kPa,
                    state,
                    posSteps,
                    DIR_SIGN
                ) = parts

                self.latest = {
                    "timestamp_ms": timestamp_ms,
                    "P_abs": P_abs,
                    "P_g": P_g,
                    "P_atm": P_atm,
                    "target_gauge_kPa": target_gauge_kPa,
                    "state": state,
                    "posSteps": posSteps,
                    "DIR_SIGN": DIR_SIGN,
                }

                self.writer.writerow([
                    time.time(),
                    timestamp_ms,
                    P_abs,
                    P_g,
                    P_atm,
                    target_gauge_kPa,
                    state,
                    posSteps,
                    DIR_SIGN
                ])
                self.csv_file.flush()

            except Exception as e:
                print("Serial read error:", e)

    def update_gui(self):
        try:
            self.pressure_label.config(text=f"P_g: {float(self.latest['P_g']):.1f} kPa")
            self.abs_label.config(text=f"P_abs: {float(self.latest['P_abs']):.1f} kPa")
            self.atm_label.config(text=f"P_atm: {float(self.latest['P_atm']):.1f} kPa")
            self.target_label.config(
                text=f"Target gauge: {float(self.latest['target_gauge_kPa']):.1f} kPa"
            )
            self.state_label.config(text=f"State: {self.latest['state']}")
        except:
            pass

        if self.running:
            self.root.after(100, self.update_gui)

    def close(self):
        self.running = False
        try:
            self.send("STOP")
            time.sleep(0.2)
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