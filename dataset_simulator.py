"""
=============================================================
  IoT Smart Study Environment — Dataset-Backed Simulator
  Project by: Bhashkar Datta Chaudhuri

  PURPOSE:
    This replaces pure synthetic simulation with REAL dataset
    values. Two public datasets are used:

    1. ENVIRONMENTAL DATA:
       "Room Occupancy Detection (IoT Sensor)" — Kaggle
       Columns: Temperature, Humidity, Light, CO2
       Source: kaggle.com/datasets/kukuroo3/room-occupancy-detection-data-iot-sensor

    2. BIOMETRIC DATA (STRESS STATES):
       "WESAD — Wearable Stress and Affect Detection" — UCI ML Repo
       Contains: BVP (→ HR), EDA, Temperature, Stress Labels
       Source: archive.ics.uci.edu/dataset/465

    The script:
    ─ Reads both datasets row by row (at 1 Hz)
    ─ Computes HR, HRV, SpO2 from WESAD stress labels
    ─ Publishes all values to local MQTT broker
    ─ Falls back to synthetic simulation if CSVs not found

  SETUP (run once):
    pip install paho-mqtt pandas numpy scipy ucimlrepo

  DOWNLOAD DATASETS:
    1. Room Occupancy CSV:
       Go to: kaggle.com/datasets/kukuroo3/room-occupancy-detection-data-iot-sensor
       Download and place as:  data/room_occupancy.csv

    2. WESAD is auto-downloaded by this script via ucimlrepo API.
       (needs pip install ucimlrepo)

  RUN:
    python dataset_simulator.py
    python dataset_simulator.py --loop         # loop dataset forever
    python dataset_simulator.py --synthetic    # use synthetic only
=============================================================
"""

import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
import json, time, argparse, os, random, math
from datetime import datetime

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────

BROKER      = "localhost"
PORT        = 1883
CLIENT_ID   = "SmartStudy_DatasetSim"

DATA_DIR         = "data"
ENV_CSV          = os.path.join(DATA_DIR, "room_occupancy.csv")
WESAD_CSV        = os.path.join(DATA_DIR, "wesad_processed.csv")

# MQTT Topics
TOPIC = {
    "co2"        : "/smartdesk/env/co2",
    "temperature": "/smartdesk/env/temperature",
    "humidity"   : "/smartdesk/env/humidity",
    "lux"        : "/smartdesk/env/lux",
    "posture"    : "/smartdesk/behavior/posture",
    "hr"         : "/smartdesk/biometrics/hr",
    "spo2"       : "/smartdesk/biometrics/spo2",
    "hrv"        : "/smartdesk/biometrics/hrv",
    "state"      : "/smartdesk/state/student",
    "actuators"  : "/smartdesk/actuators/command",
    "alert"      : "/smartdesk/alerts/notification",
}

# Thresholds (from project spec)
TH = {
    "co2_high"  : 1000,  # ppm
    "lux_low"   : 300,   # lux
    "pos_bad"   : 35,    # cm
    "hrv_stress": 25,    # ms
    "hr_stress" : 90,    # bpm
    "fan_normal": 30,    # %
    "fan_high"  : 80,    # %
    "led_warm"  : 2700,  # K
    "led_cool"  : 5000,  # K
}

# WESAD label definitions
# 1 = Baseline (normal), 2 = Stress, 3 = Amusement (relaxed)
WESAD_LABELS = {1: "BASELINE", 2: "STRESS", 3: "AMUSEMENT"}

# ─────────────────────────────────────────────────────────────
#  COLOURS (terminal output)
# ─────────────────────────────────────────────────────────────
R="\033[91m"; Y="\033[93m"; G="\033[92m"
C="\033[96m"; B="\033[94m"; W="\033[0m"
BOLD="\033[1m"; DIM="\033[2m"


# ═════════════════════════════════════════════════════════════
#  DATASET LOADER
# ═════════════════════════════════════════════════════════════

class DatasetLoader:
    """
    Loads and prepares the two datasets.
    Falls back to synthetic generation if datasets not found.
    """

    def __init__(self):
        self.env_df    = None
        self.wesad_df  = None
        self.use_real  = False
        os.makedirs(DATA_DIR, exist_ok=True)

    # ── Step 1: Load Environmental Dataset (Room Occupancy) ──

    def load_env_dataset(self):
        """
        Loads the Kaggle Room Occupancy IoT Sensor dataset.
        Expected columns: Temperature, Humidity, Light, CO2
        """
        if not os.path.exists(ENV_CSV):
            print(f"\n{Y}  ⚠ Room Occupancy CSV not found at: {ENV_CSV}{W}")
            print(f"  → Download from: kaggle.com/datasets/kukuroo3/room-occupancy-detection-data-iot-sensor")
            print(f"  → Save the file as: {ENV_CSV}")
            print(f"  → Falling back to synthetic environmental data.\n")
            return False

        try:
            df = pd.read_csv(ENV_CSV)
            print(f"\n{G}  ✅ Loaded environmental dataset: {ENV_CSV}{W}")
            print(f"     Shape: {df.shape}   Columns: {list(df.columns)}")

            # Normalise column names (dataset may use different capitalisation)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            # Map to expected names
            col_map = {
                "temperature" : "temp",
                "temp"        : "temp",
                "humidity"    : "humid",
                "light"       : "lux",
                "co2"         : "co2",
                "humidityratiohumidity_ratio": "humid_ratio",
            }
            df.rename(columns=col_map, inplace=True)

            # Keep only the columns we need
            keep = [c for c in ["temp", "humid", "lux", "co2"] if c in df.columns]
            df = df[keep].dropna()
            df = df.reset_index(drop=True)

            # Sanity check: CO2 should be in ppm range (400–5000)
            if "co2" in df.columns:
                mean_co2 = df["co2"].mean()
                if mean_co2 < 10:              # might be in ppm/1000
                    df["co2"] = df["co2"] * 1000
                print(f"     CO2 range: {df['co2'].min():.0f} – {df['co2'].max():.0f} ppm")
                print(f"     Lux range: {df['lux'].min():.0f} – {df['lux'].max():.0f}")
                print(f"     Temp range: {df['temp'].min():.1f} – {df['temp'].max():.1f} °C")

            self.env_df = df
            return True

        except Exception as e:
            print(f"{R}  ❌ Error loading environmental dataset: {e}{W}")
            return False

    # ── Step 2: Load WESAD Dataset ───────────────────────────

    def load_wesad_dataset(self):
        """
        Loads WESAD dataset. Two methods:
        Method A: Pre-processed CSV (if you already ran the extractor)
        Method B: Download via ucimlrepo API (auto)
        """
        # Method A: Load pre-processed CSV
        if os.path.exists(WESAD_CSV):
            print(f"\n{G}  ✅ Loaded WESAD processed CSV: {WESAD_CSV}{W}")
            self.wesad_df = pd.read_csv(WESAD_CSV)
            print(f"     Shape: {self.wesad_df.shape}")
            return True

        # Method B: Auto-download via ucimlrepo
        print(f"\n{C}  Attempting to download WESAD from UCI ML Repo...{W}")
        try:
            from ucimlrepo import fetch_ucirepo
            print("  Downloading WESAD (this may take a few minutes)...")
            wesad = fetch_ucirepo(id=465)
            X = wesad.data.features
            y = wesad.data.targets

            # WESAD features include: chest_ACC_x/y/z, chest_ECG, chest_EDA,
            # chest_EMG, chest_Resp, chest_Temp, wrist_ACC_x/y/z, wrist_BVP,
            # wrist_EDA, wrist_TEMP  + label column

            df = X.copy()
            if y is not None and len(y) > 0:
                df["label"] = y.values if hasattr(y, "values") else y

            # Save for future use
            df.to_csv(WESAD_CSV, index=False)
            print(f"{G}  ✅ WESAD downloaded and saved to {WESAD_CSV}{W}")
            print(f"     Shape: {df.shape}   Columns (first 8): {list(df.columns[:8])}")
            self.wesad_df = df
            return True

        except ImportError:
            print(f"{Y}  ucimlrepo not installed. Run: pip install ucimlrepo{W}")
        except Exception as e:
            print(f"{R}  ❌ Error downloading WESAD: {e}{W}")

        print(f"  → Falling back to label-based biometric simulation.\n")
        return False

    # ── Step 3: Pre-process WESAD into HR / HRV / State ──────

    def extract_biometrics_from_wesad(self):
        """
        Derives HR, HRV and stress labels from WESAD features.

        Strategy:
        - If wrist_BVP column exists → compute HR from peak intervals
        - If chest_ECG exists → compute HR from R-peak detection
        - Else → estimate from label values using physiological ranges

        Label mapping:
          1 (Baseline)  → HR 65-75 bpm, HRV 40-55 ms  → NORMAL / FOCUSED
          2 (Stress)    → HR 85-105 bpm, HRV 12-28 ms → STRESSED
          3 (Amusement) → HR 70-80 bpm, HRV 35-50 ms  → NORMAL
        """
        if self.wesad_df is None:
            return None

        df = self.wesad_df.copy()
        results = []

        # Check what columns are available
        cols = [c.lower() for c in df.columns]
        has_bvp = any("bvp" in c for c in cols)
        has_ecg = any("ecg" in c for c in cols)
        has_label = "label" in cols or "label" in df.columns.str.lower().tolist()

        print(f"\n  WESAD columns detected: BVP={has_bvp}, ECG={has_ecg}, Label={has_label}")

        if has_bvp and has_label:
            print("  Computing HR from BVP signal + stress states from labels...")
            bvp_col   = [c for c in df.columns if "bvp" in c.lower()][0]
            label_col = [c for c in df.columns if "label" in c.lower()][0]

            bvp    = df[bvp_col].values.astype(float)
            labels = df[label_col].values

            # ── BVP → HR using sliding window (64 Hz sampling rate) ──
            # WESAD wrist BVP is sampled at 64 Hz
            SAMPLE_RATE = 64
            WINDOW      = SAMPLE_RATE * 30   # 30-second windows → 1 HR value

            for i in range(0, len(bvp) - WINDOW, WINDOW // 2):   # 50% overlap
                window_bvp   = bvp[i : i + WINDOW]
                window_label = labels[i : i + WINDOW]

                # Remove NaN
                valid = ~np.isnan(window_bvp)
                if valid.sum() < WINDOW // 2:
                    continue

                bvp_clean = window_bvp[valid]

                # Simple peak detection: values above mean + 0.5*std
                threshold = np.mean(bvp_clean) + 0.5 * np.std(bvp_clean)
                peaks = []
                for j in range(1, len(bvp_clean) - 1):
                    if bvp_clean[j] > threshold and bvp_clean[j] > bvp_clean[j-1] and bvp_clean[j] > bvp_clean[j+1]:
                        if not peaks or j - peaks[-1] > SAMPLE_RATE * 0.4:  # min 0.4s between beats
                            peaks.append(j)

                if len(peaks) < 3:
                    continue

                # HR from peak count
                duration_sec = len(bvp_clean) / SAMPLE_RATE
                hr = round((len(peaks) / duration_sec) * 60, 1)

                # HRV (RMSSD) from inter-peak intervals
                rr_intervals = np.diff(peaks) / SAMPLE_RATE * 1000   # ms
                if len(rr_intervals) > 1:
                    hrv = round(np.sqrt(np.mean(np.diff(rr_intervals) ** 2)), 1)
                else:
                    hrv = 40.0

                label = int(round(np.nanmean(window_label))) if not all(np.isnan(window_label.astype(float))) else 1

                results.append({"hr": hr, "hrv": hrv, "wesad_label": label})

        elif has_label:
            # Estimate biometrics from labels alone
            print("  Estimating biometrics from WESAD stress labels...")
            label_col = [c for c in df.columns if "label" in c.lower()][0]
            labels = df[label_col].dropna().values

            # Downsample: take 1 sample per 64 rows (≈ 1 Hz from 64 Hz data)
            for i in range(0, len(labels), 64):
                lbl = int(round(labels[i]))
                if lbl == 1:   # Baseline
                    hr  = round(np.random.normal(70, 4), 1)
                    hrv = round(np.random.normal(48, 6), 1)
                elif lbl == 2:  # Stress
                    hr  = round(np.random.normal(92, 8), 1)
                    hrv = round(np.random.normal(20, 5), 1)
                elif lbl == 3:  # Amusement
                    hr  = round(np.random.normal(76, 5), 1)
                    hrv = round(np.random.normal(42, 7), 1)
                else:
                    continue
                hr  = float(np.clip(hr, 50, 145))
                hrv = float(np.clip(hrv, 8, 80))
                results.append({"hr": hr, "hrv": hrv, "wesad_label": lbl})

        if results:
            bio_df = pd.DataFrame(results)
            print(f"{G}  ✅ Biometric extraction complete: {len(bio_df)} records{W}")
            print(f"     HR  : {bio_df['hr'].min():.0f} – {bio_df['hr'].max():.0f} bpm")
            print(f"     HRV : {bio_df['hrv'].min():.0f} – {bio_df['hrv'].max():.0f} ms")
            return bio_df

        return None

    # ── Load everything ───────────────────────────────────────

    def load_all(self):
        print(f"\n{BOLD}{C}{'─'*60}")
        print("  Loading Datasets...")
        print(f"{'─'*60}{W}")
        env_ok   = self.load_env_dataset()
        wesad_ok = self.load_wesad_dataset()
        bio_df   = self.extract_biometrics_from_wesad() if wesad_ok else None
        self.use_real = env_ok or (bio_df is not None)
        return self.env_df, bio_df


# ═════════════════════════════════════════════════════════════
#  SIMULATION ENGINE
# ═════════════════════════════════════════════════════════════

class SimulationEngine:
    """
    Merges real dataset values with simulated posture,
    applies threshold logic, computes actuator commands,
    and publishes everything to MQTT.
    """

    def __init__(self, env_df, bio_df, loop=True):
        self.env_df  = env_df
        self.bio_df  = bio_df
        self.loop    = loop
        self.tick    = 0
        self.env_idx = 0
        self.bio_idx = 0

        # Posture simulation (no dataset available for this)
        self.posture_cm = 55.0

        # Fallback synthetic state
        self.syn_co2    = 450.0
        self.syn_hr     = 72.0
        self.syn_hrv    = 44.0
        self.fatigue    = 0.0
        self.stress     = 0.0

    # ── Get next environmental row ────────────────────────────

    def next_env(self):
        if self.env_df is not None and len(self.env_df) > 0:
            row = self.env_df.iloc[self.env_idx]
            self.env_idx += 1
            if self.env_idx >= len(self.env_df):
                if self.loop:
                    self.env_idx = 0
                    print(f"\n{C}  [Dataset] Environmental data looped back to start.{W}")
                else:
                    return None
            return {
                "co2"  : float(row.get("co2",   450)),
                "temp" : float(row.get("temp",  26.5)),
                "humid": float(row.get("humid", 63.0)),
                "lux"  : float(row.get("lux",   400)),
            }
        # Synthetic fallback
        self.syn_co2 += 0.3 + self.fatigue * 0.4 + random.gauss(0, 5)
        self.syn_co2  = max(400, min(1600, self.syn_co2))
        lux = max(80, min(1200, 420 + math.sin(self.tick / 120) * 30 + random.gauss(0, 8)))
        return {
            "co2"  : round(self.syn_co2, 1),
            "temp" : round(26.5 + random.gauss(0, 0.1), 1),
            "humid": round(max(40, min(85, 63 + random.gauss(0, 0.2))), 1),
            "lux"  : round(lux, 1),
        }

    # ── Get next biometric row ────────────────────────────────

    def next_bio(self):
        if self.bio_df is not None and len(self.bio_df) > 0:
            row = self.bio_df.iloc[self.bio_idx]
            self.bio_idx += 1
            if self.bio_idx >= len(self.bio_df):
                if self.loop:
                    self.bio_idx = 0
                    print(f"\n{C}  [Dataset] Biometric data looped back to start.{W}")
                else:
                    return None
            hr  = float(row.get("hr",  72))
            hrv = float(row.get("hrv", 44))
            lbl = int(row.get("wesad_label", 1))
            # SpO2 derived from CO2 and HRV
            spo2 = round(max(91, min(100, 98.5 - (lbl == 2) * 1.5 + random.gauss(0, 0.2))), 1)
            return {"hr": hr, "hrv": hrv, "spo2": spo2, "wesad_label": lbl}

        # Synthetic fallback
        self.fatigue = min(1.0, self.fatigue + 1/3600)
        self.stress  = max(0, self.stress - 0.003)
        if random.random() < 0.008:
            self.stress = min(1.0, self.stress + random.uniform(0.1, 0.25))
        hr_target = 72 + self.stress * 28 + self.fatigue * 8
        self.syn_hr  = self.syn_hr * 0.97 + hr_target * 0.03 + random.gauss(0, 1.2)
        self.syn_hr  = max(50, min(145, self.syn_hr))
        hrv_target = 50 - self.stress * 35 - self.fatigue * 10
        self.syn_hrv = self.syn_hrv * 0.96 + hrv_target * 0.04 + random.gauss(0, 2)
        self.syn_hrv = max(8, min(80, self.syn_hrv))
        spo2 = round(max(91, min(100, 98.5 - max(0, (self.syn_co2 - 800)/1000)*2.5 + random.gauss(0, 0.2))), 1)
        return {"hr": round(self.syn_hr, 1), "hrv": round(self.syn_hrv, 1), "spo2": spo2, "wesad_label": 1}

    # ── Simulate posture (HC-SR04) ────────────────────────────

    def next_posture(self, fatigue_proxy=0.0):
        target = 56 - fatigue_proxy * 20
        self.posture_cm = self.posture_cm * 0.97 + target * 0.03
        self.posture_cm += random.gauss(0, 1.5)
        if random.random() < 0.03:
            self.posture_cm -= random.uniform(8, 20)  # sudden lean
        self.posture_cm = max(15, min(90, self.posture_cm))
        return round(self.posture_cm, 1)

    # ── Infer student state ───────────────────────────────────

    def infer_state(self, hr, hrv, co2, posture, wesad_label=1):
        """
        Combines WESAD label with threshold checks.
        WESAD label 2 = confirmed stress from real physiological data.
        """
        if wesad_label == 2:     # Real stress data from WESAD
            return "STRESSED"
        if hrv < TH["hrv_stress"] and hr > TH["hr_stress"]:
            return "STRESSED"
        if co2 > TH["co2_high"] or posture < TH["pos_bad"]:
            return "FATIGUED"
        if hr <= 80 and hrv >= 35 and posture >= 40:
            return "FOCUSED"
        return "NORMAL"

    # ── Compute actuator commands ─────────────────────────────

    def compute_actuators(self, state, co2, lux, posture):
        fan      = TH["fan_normal"]
        led_brt  = 70
        led_k    = TH["led_cool"]
        haptic   = False
        oled     = "✅ All Good"
        brk      = False

        if co2 > TH["co2_high"]:
            fan  = TH["fan_high"]
            oled = "⚠️ Ventilation Required"

        if lux < TH["lux_low"]:
            led_brt = min(100, int(70 + (500 - lux) / 5))

        if posture < TH["pos_bad"]:
            haptic = True
            oled   = "📐 Sit Straight!"

        if state == "STRESSED":
            led_k = TH["led_warm"]
            brk   = True
            oled  = "😮‍💨 Take a 5-min Break"

        return {
            "fan_pwm"        : fan,
            "led_brightness" : led_brt,
            "led_color_k"    : led_k,
            "haptic"         : haptic,
            "oled_message"   : oled,
            "break_suggested": brk,
        }


# ═════════════════════════════════════════════════════════════
#  MQTT CLIENT
# ═════════════════════════════════════════════════════════════

def build_mqtt_client():
    client = mqtt.Client(CLIENT_ID)
    def on_connect(cl, ud, flags, rc):
        if rc == 0:
            print(f"\n{G}  ✅ MQTT Connected to {BROKER}:{PORT}{W}")
        else:
            print(f"{R}  ❌ MQTT connection failed (code {rc}){W}")
            print("  → Make sure Mosquitto is running: mosquitto -v")
    client.on_connect = on_connect
    return client


# ═════════════════════════════════════════════════════════════
#  MAIN SIMULATION LOOP
# ═════════════════════════════════════════════════════════════

def run(loop_data=True, use_synthetic=False):
    print(f"""
{BOLD}{C}{'═'*62}
  IoT Smart Study Environment
  Dataset-Backed MQTT Simulator
{'═'*62}{W}""")

    # ── Load datasets ──
    loader = DatasetLoader()
    if use_synthetic:
        env_df, bio_df = None, None
        print(f"\n{Y}  [--synthetic] Using synthetic data only.{W}")
    else:
        env_df, bio_df = loader.load_all()

    # ── Connect MQTT ──
    client = build_mqtt_client()
    try:
        client.connect(BROKER, PORT, keepalive=60)
        client.loop_start()
        time.sleep(1)
    except ConnectionRefusedError:
        print(f"\n{R}  ❌ Cannot connect to MQTT broker!{W}")
        print("  Start Mosquitto first:")
        print("    Windows → open CMD: mosquitto -v")
        print("    Linux   → sudo systemctl start mosquitto")
        return

    # ── Create simulation engine ──
    engine = SimulationEngine(env_df, bio_df, loop=loop_data)

    print(f"""
{BOLD}{'─'*62}
  Simulation started. Press Ctrl+C to stop.
  Open Node-RED dashboard: http://localhost:1880/ui
{'─'*62}{W}""")

    # Tracking for fatigue proxy
    fatigue_proxy = 0.0
    alert_cooldowns = {}   # prevent spam alerts

    try:
        while True:
            engine.tick += 1
            fatigue_proxy = min(1.0, fatigue_proxy + 1/3600)

            # ── Read all sensors ──
            env  = engine.next_env()
            bio  = engine.next_bio()
            pos  = engine.next_posture(fatigue_proxy)

            if env is None or bio is None:
                print("\n  Dataset exhausted. Stopping.")
                break

            # ── Add noise to dataset values (simulate sensor imprecision) ──
            co2    = round(float(env["co2"])   + random.gauss(0, 8),  1)
            temp   = round(float(env["temp"])  + random.gauss(0, 0.1),1)
            humid  = round(float(env["humid"]) + random.gauss(0, 0.5),1)
            lux    = round(float(env["lux"])   + random.gauss(0, 6),  1)
            hr     = round(float(bio["hr"])    + random.gauss(0, 1.5),1)
            hrv    = round(float(bio["hrv"])   + random.gauss(0, 1.5),1)
            spo2   = round(float(bio["spo2"]), 1)
            wlbl   = int(bio["wesad_label"])

            # Clamp all values
            co2   = max(400,  min(2000, co2))
            lux   = max(50,   min(1500, lux))
            hr    = max(45,   min(145,  hr))
            hrv   = max(5,    min(90,   hrv))
            spo2  = max(90,   min(100,  spo2))

            # ── Infer state and actuators ──
            state = engine.infer_state(hr, hrv, co2, pos, wlbl)
            cmds  = engine.compute_actuators(state, co2, lux, pos)

            # ── MQTT Payloads ──
            ts = datetime.now().isoformat()
            payloads = {
                TOPIC["co2"]        : {"value": co2,   "unit": "ppm",  "source": "dataset", "ts": ts},
                TOPIC["temperature"]: {"value": temp,  "unit": "°C",   "source": "dataset", "ts": ts},
                TOPIC["humidity"]   : {"value": humid, "unit": "%",    "source": "dataset", "ts": ts},
                TOPIC["lux"]        : {"value": lux,   "unit": "lux",  "source": "dataset", "ts": ts},
                TOPIC["posture"]    : {"value": pos,   "unit": "cm",   "source": "simulated","ts": ts},
                TOPIC["hr"]         : {"value": hr,    "unit": "bpm",  "source": "WESAD",   "ts": ts},
                TOPIC["hrv"]        : {"value": hrv,   "unit": "ms",   "source": "WESAD",   "ts": ts},
                TOPIC["spo2"]       : {"value": spo2,  "unit": "%",    "source": "derived", "ts": ts},
                TOPIC["state"]      : {
                    "state"      : state,
                    "wesad_label": WESAD_LABELS.get(wlbl, "UNKNOWN"),
                    "fatigue"    : round(fatigue_proxy, 3),
                    "ts"         : ts
                },
                TOPIC["actuators"]  : {**cmds, "ts": ts},
            }

            for topic, payload in payloads.items():
                client.publish(topic, json.dumps(payload), qos=1, retain=True)

            # ── Publish alerts (with cooldown to avoid spam) ──
            def alert(key, msg, cooldown_sec=30):
                last = alert_cooldowns.get(key, 0)
                if engine.tick - last >= cooldown_sec:
                    alert_cooldowns[key] = engine.tick
                    client.publish(TOPIC["alert"], json.dumps({"type": key, "message": msg, "ts": ts}), qos=1)
                    return True
                return False

            if co2 > TH["co2_high"]: alert("co2",     f"HIGH CO₂ ({co2:.0f} ppm) → Fans at 80% PWM")
            if lux < TH["lux_low"]:  alert("light",   f"LOW LIGHT ({lux:.0f} lux) → LED Brightness Boosted")
            if pos < TH["pos_bad"]:  alert("posture",  f"POOR POSTURE ({pos:.0f} cm) → Haptic Alert Active")
            if state == "STRESSED":  alert("stress",   f"STRESS DETECTED (WESAD Label={wlbl}) HR={hr:.0f} HRV={hrv:.0f}")

            # ── Terminal output ──
            elapsed = f"{engine.tick//60:02d}:{engine.tick%60:02d}"
            state_col = {
                "FOCUSED":"🟢"+G, "NORMAL":"🔵"+B,
                "FATIGUED":"🟡"+Y, "STRESSED":"🔴"+R
            }.get(state, "⚫")

            print(f"\n{DIM}{'─'*62}{W}")
            print(f"  {BOLD}[{elapsed}] #{engine.tick:4d} | WESAD: {WESAD_LABELS.get(wlbl,'?')} | State: {state_col}{state}{W}")
            print(f"  {BOLD}ENV  :{W} CO2={co2:.0f}ppm  Lux={lux:.0f}lx  Temp={temp:.1f}°C  Humid={humid:.0f}%")
            print(f"  {BOLD}BIO  :{W} HR={hr:.0f}bpm  HRV={hrv:.0f}ms  SpO2={spo2:.1f}%  [source: WESAD]")
            print(f"  {BOLD}POSTURE:{W} {pos:.0f}cm  [source: simulated HC-SR04]")
            print(f"  {BOLD}ACTUAT:{W} Fan={cmds['fan_pwm']}%  LED={cmds['led_brightness']}%@{cmds['led_color_k']}K  Haptic={'ON' if cmds['haptic'] else 'off'}  Break={cmds['break_suggested']}")
            print(f"  {BOLD}OLED  :{W} {cmds['oled_message']}")

            time.sleep(1.0)   # 1 Hz polling

    except KeyboardInterrupt:
        print(f"\n\n{C}  Stopped. Total ticks: {engine.tick}{W}")
    finally:
        client.loop_stop()
        client.disconnect()


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT Smart Study Environment — Dataset Simulator")
    parser.add_argument("--loop",      action="store_true", default=True,  help="Loop dataset when exhausted")
    parser.add_argument("--no-loop",   action="store_true", default=False, help="Stop when dataset exhausted")
    parser.add_argument("--synthetic", action="store_true", default=False, help="Use synthetic data only")
    args = parser.parse_args()
    run(loop_data=not args.no_loop, use_synthetic=args.synthetic)
