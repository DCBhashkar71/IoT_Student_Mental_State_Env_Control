"""
=============================================================
  IoT Smart Study Environment - Sensor Simulator
  Project by: Bhashkar Datta Chaudhuri
  
  Description:
    This script simulates all physical sensors of the Smart
    Study Desk and publishes data to a local MQTT broker
    (Mosquitto) at 1Hz polling rate, exactly as the ESP32
    would do in real hardware.
    
  Sensors Simulated:
    - CCS811  → CO2 (ppm) + TVOC
    - DHT22   → Temperature (°C) + Humidity (%)
    - BH1750  → Ambient Light (lux)
    - HC-SR04 → Posture Distance (cm)
    - MAX30102 → Heart Rate (bpm) + SpO2 (%) + HRV (ms)
    
  MQTT Topics Published:
    /smartdesk/env/co2
    /smartdesk/env/temperature
    /smartdesk/env/humidity
    /smartdesk/env/lux
    /smartdesk/behavior/posture
    /smartdesk/biometrics/hr
    /smartdesk/biometrics/spo2
    /smartdesk/biometrics/hrv
    /smartdesk/state/student
    /smartdesk/actuators/command   ← actuator decisions

  Requirements:
    pip install paho-mqtt
    
  Run:
    python sensor_simulator.py
    
  Make sure Mosquitto is running first:
    Windows: mosquitto -v
    Linux/Mac: sudo systemctl start mosquitto
=============================================================
"""

import paho.mqtt.client as mqtt
import time
import random
import math
import json
import argparse
from datetime import datetime


# ─────────────────────────────────────────────
#  MQTT CONFIGURATION
# ─────────────────────────────────────────────
BROKER_HOST = "localhost"
BROKER_PORT = 1883
CLIENT_ID   = "SmartStudySimulator_v1"
POLL_RATE_HZ = 1          # 1 reading per second (as per hardware spec)

# ─────────────────────────────────────────────
#  MQTT TOPIC MAP  (matches Node-RED flow)
# ─────────────────────────────────────────────
TOPIC = {
    "co2"         : "/smartdesk/env/co2",
    "temperature" : "/smartdesk/env/temperature",
    "humidity"    : "/smartdesk/env/humidity",
    "lux"         : "/smartdesk/env/lux",
    "posture"     : "/smartdesk/behavior/posture",
    "hr"          : "/smartdesk/biometrics/hr",
    "spo2"        : "/smartdesk/biometrics/spo2",
    "hrv"         : "/smartdesk/biometrics/hrv",
    "state"       : "/smartdesk/state/student",
    "actuators"   : "/smartdesk/actuators/command",
    "alert"       : "/smartdesk/alerts/notification",
}

# ─────────────────────────────────────────────
#  THRESHOLD CONSTANTS  (from project spec)
# ─────────────────────────────────────────────
THRESHOLD = {
    "co2_fatigue"     : 1000,   # ppm  → trigger fans
    "lux_minimum"     : 300,    # lux  → boost LED
    "lux_target"      : 500,    # lux  → LED target
    "posture_bad"     : 35,     # cm   → too close (slouching)
    "posture_ok"      : 55,     # cm   → normal distance
    "hrv_stress"      : 25,     # ms   → low HRV = stressed
    "hr_stress"       : 90,     # bpm  → elevated HR
    "fan_pwm_normal"  : 30,     # %    → background fan
    "fan_pwm_high"    : 80,     # %    → high CO2 response
    "led_warm_k"      : 2700,   # K    → calming amber
    "led_cool_k"      : 5000,   # K    → focus white
}

# ─────────────────────────────────────────────
#  STUDENT STATE MACHINE
# ─────────────────────────────────────────────
STATES = {
    "FOCUSED"  : "🟢 FOCUSED",
    "NORMAL"   : "🔵 NORMAL",
    "FATIGUED" : "🟡 FATIGUED",
    "STRESSED" : "🔴 STRESSED",
    "BREAK"    : "⚪ BREAK",
}


class StudentStateSimulator:
    """
    Simulates a student studying over time.
    Models gradual fatigue, CO2 buildup, and stress events
    realistically — just like a real study session would unfold.
    """

    def __init__(self, scenario="normal"):
        self.tick           = 0            # seconds elapsed
        self.scenario       = scenario     # "normal", "stress", "tired"
        self.fatigue_level  = 0.0          # 0.0 – 1.0
        self.stress_level   = 0.0          # 0.0 – 1.0
        self.is_on_break    = False
        self.break_timer    = 0

        # Baseline sensor values (Chennai environment)
        self._co2_ppm       = 450.0
        self._temp_c        = 26.5
        self._humidity_pct  = 62.0
        self._lux           = 420.0
        self._posture_cm    = 56.0
        self._hr_bpm        = 72.0
        self._spo2_pct      = 98.5
        self._hrv_ms        = 44.0

        # Noise seeds  (smooth random walk)
        self._co2_noise     = 0.0
        self._lux_noise     = 0.0
        self._posture_noise = 0.0
        self._hr_noise      = 0.0

        # Apply scenario starting conditions
        # Apply scenario starting conditions
        if scenario == "stress":
            self.stress_level   = 0.9       # very high stress from start
            self._hr_bpm        = 97.0      # clearly above 90 threshold
            self._hrv_ms        = 16.0      # clearly below 25 threshold
            self._co2_ppm       = 680.0     # slightly elevated
            self._posture_cm    = 44.0      # acceptable posture

        elif scenario == "tired":
            self.fatigue_level  = 0.85      # heavily fatigued
            self._co2_ppm       = 1080.0    # already above 1000 threshold
            self._posture_cm    = 28.0      # already below 35cm threshold
            self._hr_bpm        = 80.0      # slightly elevated
            self._hrv_ms        = 30.0      # borderline

    # ── Internal helpers ──────────────────────────────────────

    def _smooth_noise(self, current, target_range, speed=0.1):
        """Smooth random walk for realistic sensor fluctuation."""
        drift = random.gauss(0, target_range * 0.3)
        return current * (1 - speed) + (current + drift) * speed

    def _clamp(self, value, lo, hi):
        return max(lo, min(hi, value))

    # ── Update simulation state ───────────────────────────────

    def tick_update(self):
        self.tick += 1

        # ── Fatigue builds up over time (max after ~1 hour) ──
        fatigue_rate = 1 / 3600 if self.scenario != "tired" else 1 / 1800
        self.fatigue_level = self._clamp(self.fatigue_level + fatigue_rate, 0, 1)

        # ── CO2 drifts up as student breathes in closed room ──
        co2_drift = 0.3 + self.fatigue_level * 0.4   # faster buildup when tired
        self._co2_ppm += co2_drift + random.gauss(0, 5)
        self._co2_ppm = self._clamp(self._co2_ppm, 400, 1800)

        # ── Random stress spikes (like hard problems, deadlines) ──
        if random.random() < 0.008:   # ~0.8% chance per second
            self.stress_level = self._clamp(self.stress_level + random.uniform(0.1, 0.3), 0, 1)
        # Stress naturally decays
        self.stress_level = self._clamp(self.stress_level - 0.003, 0, 1)

        # ── Break logic: auto-break after 25 minutes (Pomodoro) ──
        if self.tick % 1500 == 0 and not self.is_on_break:
            self.is_on_break = True
            self.break_timer = 300   # 5-minute break

        if self.is_on_break:
            self.break_timer -= 1
            # During break: CO2 drops, stress drops, posture normalises
            self._co2_ppm  = max(400, self._co2_ppm - 2)
            self.stress_level = max(0, self.stress_level - 0.01)
            self.fatigue_level = max(0, self.fatigue_level - 0.002)
            if self.break_timer <= 0:
                self.is_on_break = False

        # ── Light: simulate cloud passing / sunlight change ──
        self._lux += math.sin(self.tick / 120) * 2 + random.gauss(0, 8)
        self._lux = self._clamp(self._lux, 80, 1200)

        # ── Posture: gets worse as fatigue grows ──
        posture_target = 56 - (self.fatigue_level * 20) - (self.stress_level * 5)
        self._posture_cm = self._posture_cm * 0.98 + posture_target * 0.02
        self._posture_cm += random.gauss(0, 1.5)
        self._posture_cm = self._clamp(self._posture_cm, 15, 90)

        # ── Heart rate: rises with stress + light fatigue effect ──
        hr_target = 72 + self.stress_level * 28 + self.fatigue_level * 8
        self._hr_bpm = self._hr_bpm * 0.97 + hr_target * 0.03
        self._hr_bpm += random.gauss(0, 1.2)
        self._hr_bpm = self._clamp(self._hr_bpm, 50, 145)

        # ── HRV: inversely correlated with stress ──
        hrv_target = 50 - self.stress_level * 35 - self.fatigue_level * 10
        self._hrv_ms = self._hrv_ms * 0.96 + hrv_target * 0.04
        self._hrv_ms += random.gauss(0, 2)
        self._hrv_ms = self._clamp(self._hrv_ms, 8, 85)

        # ── SpO2: drops slightly if CO2 is very high ──
        co2_effect = max(0, (self._co2_ppm - 800) / 1000) * 2.5
        self._spo2_pct = 98.5 - co2_effect + random.gauss(0, 0.2)
        self._spo2_pct = self._clamp(self._spo2_pct, 91, 100)

        # ── Temperature / Humidity: slow drift ──
        self._temp_c += random.gauss(0, 0.05)
        self._temp_c = self._clamp(self._temp_c, 24, 32)
        self._humidity_pct += random.gauss(0, 0.2)
        self._humidity_pct = self._clamp(self._humidity_pct, 40, 85)

    # ── Sensor read methods ────────────────────────────────────

    def read_co2(self):
        return round(self._co2_ppm, 1)

    def read_temperature(self):
        return round(self._temp_c, 1)

    def read_humidity(self):
        return round(self._humidity_pct, 1)

    def read_lux(self):
        return round(self._lux, 1)

    def read_posture(self):
        return round(self._posture_cm, 1)

    def read_heart_rate(self):
        return int(round(self._hr_bpm))

    def read_spo2(self):
        return round(self._spo2_pct, 1)

    def read_hrv(self):
        return round(self._hrv_ms, 1)

    # ── State inference (mirrors Node-RED logic) ──────────────

    def infer_student_state(self):
        if self.is_on_break:
            return "BREAK"
        hr  = self.read_heart_rate()
        hrv = self.read_hrv()
        co2 = self.read_co2()
        pos = self.read_posture()

        if hrv < THRESHOLD["hrv_stress"] and hr > THRESHOLD["hr_stress"]:
            return "STRESSED"
        elif co2 > THRESHOLD["co2_fatigue"] or pos < THRESHOLD["posture_bad"]:
            return "FATIGUED"
        elif hr <= 80 and hrv >= 35 and pos >= 40:
            return "FOCUSED"
        else:
            return "NORMAL"

    # ── Actuator decision engine ──────────────────────────────

    def compute_actuator_commands(self, state):
        """
        Replicates the real ESP32 actuator logic in software.
        Returns a dict of all actuator states to publish.
        """
        co2 = self.read_co2()
        lux = self.read_lux()
        pos = self.read_posture()

        commands = {
            "fan_pwm"         : THRESHOLD["fan_pwm_normal"],
            "led_brightness"  : 70,
            "led_color_k"     : THRESHOLD["led_cool_k"],
            "haptic"          : False,
            "oled_message"    : "✅ All Good",
            "break_suggested" : False,
        }

        # Rule 1: Air Quality Control
        if co2 > THRESHOLD["co2_fatigue"]:
            commands["fan_pwm"]       = THRESHOLD["fan_pwm_high"]
            commands["oled_message"]  = "⚠️ Ventilation Required"

        # Rule 2: Adaptive Lighting
        if lux < THRESHOLD["lux_minimum"]:
            # PWM brightness scaled to reach target lux
            brightness_boost = min(100, int(70 + (THRESHOLD["lux_target"] - lux) / 5))
            commands["led_brightness"] = brightness_boost

        # Rule 3: Posture Correction
        if pos < THRESHOLD["posture_bad"]:
            commands["haptic"]       = True
            commands["oled_message"] = "📐 Sit Straight!"

        # Rule 4: Stress Management
        if state == "STRESSED":
            commands["led_color_k"]     = THRESHOLD["led_warm_k"]  # calming amber
            commands["break_suggested"] = True
            commands["oled_message"]    = "😮‍💨 Take a 5-min Break"

        # Rule 5: Break mode
        if state == "BREAK":
            commands["led_color_k"]     = THRESHOLD["led_warm_k"]
            commands["led_brightness"]  = 40
            commands["fan_pwm"]         = 20
            commands["oled_message"]    = "☕ Break Time — Relax!"

        return commands


# ─────────────────────────────────────────────
#  MQTT CALLBACKS
# ─────────────────────────────────────────────

def on_connect(client, userdata, flags, rc):
    codes = {
        0: "Connected successfully ✅",
        1: "Bad protocol version",
        2: "Bad client ID",
        3: "Server unavailable — Is Mosquitto running?",
        4: "Bad username or password",
        5: "Not authorised",
    }
    print(f"\n  MQTT: {codes.get(rc, f'Unknown error code {rc}')}")
    if rc != 0:
        print("  → Start Mosquitto: run 'mosquitto -v' in a terminal")

def on_disconnect(client, userdata, rc):
    print(f"\n  MQTT disconnected (code {rc})")

def on_publish(client, userdata, mid):
    pass   # silent — we already print summary


# ─────────────────────────────────────────────
#  COLOUR HELPERS (terminal output)
# ─────────────────────────────────────────────

RESET  = "\033[0m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

def coloured_state(state):
    colors = {
        "FOCUSED"  : GREEN,
        "NORMAL"   : BLUE,
        "FATIGUED" : YELLOW,
        "STRESSED" : RED,
        "BREAK"    : CYAN,
    }
    return f"{colors.get(state, '')}{BOLD}{state}{RESET}"

def coloured_value(value, lo_warn, hi_warn, unit="", invert=False):
    """Red if outside thresholds, green if OK."""
    if invert:
        warn = value < lo_warn
    else:
        warn = value > hi_warn or value < lo_warn
    color = RED if warn else GREEN
    return f"{color}{value}{unit}{RESET}"


# ─────────────────────────────────────────────
#  MAIN SIMULATION LOOP
# ─────────────────────────────────────────────

def run_simulation(scenario="normal"):
    print(f"""
{BOLD}{CYAN}{'=' * 62}
   IoT Smart Study Environment — Sensor Simulator
   Scenario: {scenario.upper()}
{'=' * 62}{RESET}
  📡 Broker  : {BROKER_HOST}:{BROKER_PORT}
  ⏱  Rate    : {POLL_RATE_HZ} Hz  (1 reading/second)
  🛑 Stop    : Ctrl + C
{'─' * 62}""")

    # Setup MQTT
    client = mqtt.Client(CLIENT_ID)
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish    = on_publish

    print(f"\n  Connecting to MQTT broker at {BROKER_HOST}:{BROKER_PORT}...")
    try:
        client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
        client.loop_start()
        time.sleep(1)
    except ConnectionRefusedError:
        print(f"\n{RED}  ❌ Cannot connect to MQTT broker!{RESET}")
        print("  Make sure Mosquitto is running:")
        print("    Windows → open CMD and run: mosquitto -v")
        print("    Linux   → sudo systemctl start mosquitto")
        print("    Mac     → brew services start mosquitto")
        return
    except Exception as e:
        print(f"\n{RED}  ❌ Connection error: {e}{RESET}")
        return

    sim = StudentStateSimulator(scenario=scenario)

    try:
        while True:
            sim.tick_update()

            # ── Read all sensor values ──────────────────────
            co2   = sim.read_co2()
            temp  = sim.read_temperature()
            humid = sim.read_humidity()
            lux   = sim.read_lux()
            pos   = sim.read_posture()
            hr    = sim.read_heart_rate()
            spo2  = sim.read_spo2()
            hrv   = sim.read_hrv()
            state = sim.infer_student_state()
            cmds  = sim.compute_actuator_commands(state)

            ts = datetime.now().isoformat()

            # ── Build MQTT payloads (JSON) ─────────────────
            payloads = {
                TOPIC["co2"]:         {"value": co2,   "unit": "ppm",  "ts": ts},
                TOPIC["temperature"]: {"value": temp,  "unit": "°C",   "ts": ts},
                TOPIC["humidity"]:    {"value": humid, "unit": "%",    "ts": ts},
                TOPIC["lux"]:         {"value": lux,   "unit": "lux",  "ts": ts},
                TOPIC["posture"]:     {"value": pos,   "unit": "cm",   "ts": ts},
                TOPIC["hr"]:          {"value": hr,    "unit": "bpm",  "ts": ts},
                TOPIC["spo2"]:        {"value": spo2,  "unit": "%",    "ts": ts},
                TOPIC["hrv"]:         {"value": hrv,   "unit": "ms",   "ts": ts},
                TOPIC["state"]:       {"state": state, "fatigue": round(sim.fatigue_level, 3),
                                       "stress": round(sim.stress_level, 3), "ts": ts},
                TOPIC["actuators"]:   {**cmds, "ts": ts},
            }

            # ── Publish all topics ─────────────────────────
            for topic, payload in payloads.items():
                client.publish(topic, json.dumps(payload), qos=1, retain=True)

            # ── Generate alerts ────────────────────────────
            alerts = []
            if co2   > THRESHOLD["co2_fatigue"]:   alerts.append(("co2",     "HIGH CO2 — Fans at 80%"))
            if lux   < THRESHOLD["lux_minimum"]:   alerts.append(("light",   "LOW LIGHT — Boosting LED"))
            if pos   < THRESHOLD["posture_bad"]:   alerts.append(("posture", "POOR POSTURE — Haptic Alert"))
            if hrv   < THRESHOLD["hrv_stress"]:    alerts.append(("stress",  "HIGH STRESS — Break Advised"))
            if spo2  < 95:                          alerts.append(("spo2",    "LOW SpO2 — Check Breathing"))

            for alert_type, msg in alerts:
                alert_payload = {"type": alert_type, "message": msg, "ts": ts}
                client.publish(TOPIC["alert"], json.dumps(alert_payload), qos=1)

            # ── Console Dashboard ──────────────────────────
            elapsed = f"{sim.tick // 60:02d}:{sim.tick % 60:02d}"
            print(f"\n{DIM}{'─' * 62}{RESET}")
            print(f"  {BOLD}[{elapsed}] Tick #{sim.tick:4d}   State: {coloured_state(state)}{RESET}")
            print(f"\n  {BOLD}ENVIRONMENT:{RESET}")
            print(f"    CO2:         {coloured_value(co2, 400, 1000, ' ppm')}  {DIM}(fan: {cmds['fan_pwm']}% PWM){RESET}")
            print(f"    Temperature: {temp}°C    Humidity: {humid}%")
            print(f"    Light:       {coloured_value(lux, 300, 9999, ' lux')}  {DIM}(LED: {cmds['led_brightness']}% @ {cmds['led_color_k']}K){RESET}")
            print(f"\n  {BOLD}BIOMETRICS:{RESET}")
            print(f"    Heart Rate:  {coloured_value(hr, 50, 90, ' bpm')}")
            print(f"    SpO2:        {coloured_value(spo2, 95, 100, '%', invert=True)}")
            print(f"    HRV:         {coloured_value(hrv, 25, 100, ' ms', invert=True)}")
            print(f"\n  {BOLD}BEHAVIOR:{RESET}")
            print(f"    Posture:     {coloured_value(pos, 35, 100, ' cm', invert=True)}  {DIM}(from sensor){RESET}")
            print(f"    Fatigue:     {round(sim.fatigue_level * 100, 1)}%   Stress: {round(sim.stress_level * 100, 1)}%")
            print(f"\n  {BOLD}ACTUATORS:{RESET}")
            haptic_str = f"{RED}VIBRATING{RESET}" if cmds["haptic"] else f"{GREEN}OFF{RESET}"
            break_str  = f"{YELLOW}YES{RESET}" if cmds["break_suggested"] else "No"
            print(f"    Fan:         {cmds['fan_pwm']}% PWM")
            print(f"    LED:         {cmds['led_brightness']}% brightness @ {cmds['led_color_k']}K")
            print(f"    Haptic:      {haptic_str}")
            print(f"    OLED:        {cmds['oled_message']}")
            print(f"    Break?:      {break_str}")

            if alerts:
                print(f"\n  {BOLD}{RED}⚠️  ALERTS:{RESET}")
                for _, msg in alerts:
                    print(f"    {RED}→ {msg}{RESET}")

            time.sleep(1.0 / POLL_RATE_HZ)

    except KeyboardInterrupt:
        print(f"\n\n{CYAN}  Simulation stopped by user.{RESET}")
        print(f"  Total time simulated: {sim.tick} seconds ({sim.tick // 60} min {sim.tick % 60} sec)")
    finally:
        client.loop_stop()
        client.disconnect()
        print("  MQTT disconnected. Goodbye!\n")


# ─────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT Smart Study Environment — Sensor Simulator")
    parser.add_argument(
        "--scenario",
        choices=["normal", "stress", "tired"],
        default="normal",
        help="Simulation scenario (default: normal)"
    )
    args = parser.parse_args()
    run_simulation(scenario=args.scenario)
