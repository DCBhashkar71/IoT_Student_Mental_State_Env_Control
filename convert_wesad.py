"""
WESAD Dataset Converter
Bhashkar Datta Chaudhuri — IoT Smart Study Environment
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────────────────────
# PATHS — These match your folder structure exactly
# ─────────────────────────────────────────────────────────────
WESAD_FOLDER = r"C:\Users\HP\Downloads\archive\WESAD"
OUTPUT_CSV   = r"C:\SET\smart_study_env\data\wesad_processed.csv"

# All subject IDs in WESAD
SUBJECT_IDS  = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

# ─────────────────────────────────────────────────────────────

def read_e4_csv(filepath):
    """
    Reads Empatica E4 CSV format.
    Line 1 = start timestamp
    Line 2 = sample rate
    Line 3+ = actual values
    """
    if not os.path.exists(filepath):
        return None, None, []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if len(lines) < 3:
        return None, None, []

    try:
        start_time  = float(lines[0].strip())
        sample_rate = float(lines[1].strip())
        values = []
        for line in lines[2:]:
            line = line.strip()
            if line:
                try:
                    values.append(float(line))
                except:
                    continue
        return start_time, sample_rate, values
    except:
        return None, None, []


def compute_hrv_from_ibi(ibi_filepath):
    """
    Reads IBI.csv and computes HRV (RMSSD) in 30-second windows.
    IBI = Inter Beat Interval (time between heartbeats in seconds)
    RMSSD = Root Mean Square of Successive Differences
    High RMSSD = calm. Low RMSSD = stressed.
    """
    if not os.path.exists(ibi_filepath):
        return []

    with open(ibi_filepath, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        return []

    ibi_times  = []
    ibi_values = []

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 2:
            try:
                t   = float(parts[0].strip())
                ibi = float(parts[1].strip())
                ibi_times.append(t)
                ibi_values.append(ibi * 1000)  # convert seconds to ms
            except:
                continue

    if len(ibi_values) < 3:
        return []

    # Compute RMSSD in 30-second windows
    hrv_list    = []
    window_size = 30  # seconds
    i = 0

    while i < len(ibi_times):
        window_start = ibi_times[i]
        window_ibis  = []

        j = i
        while j < len(ibi_times) and (ibi_times[j] - window_start) <= window_size:
            window_ibis.append(ibi_values[j])
            j += 1

        if len(window_ibis) >= 3:
            diffs = np.diff(window_ibis)
            rmssd = np.sqrt(np.mean(diffs ** 2))
            hrv_list.append(round(float(rmssd), 2))

        i = max(i + 1, j)

    return hrv_list


def infer_label(hr, hrv):
    """
    Infers stress label from HR and HRV values.
    Based on WESAD published physiological ranges:
    Label 1 = Baseline (calm/normal studying)
    Label 2 = Stress
    Label 3 = Amusement (relaxed)
    """
    if hr > 85 and hrv < 28:
        return 2   # Stress
    elif hr < 80 and hrv > 35:
        return 1   # Baseline
    else:
        return 3   # Amusement / Neutral


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

all_records = []

print("=" * 60)
print("  WESAD Converter — IoT Smart Study Environment")
print("  Reading from:", WESAD_FOLDER)
print("=" * 60)

for sid in SUBJECT_IDS:

    subject_folder = os.path.join(WESAD_FOLDER, f"S{sid}")
    e4_folder      = os.path.join(subject_folder, f"S{sid}_E4_Data")

    if not os.path.exists(e4_folder):
        print(f"\n  S{sid}: Folder not found — skipping")
        continue

    print(f"\n  Processing Subject S{sid}...")

    # ── Read HR.csv ──────────────────────────────────────────
    hr_path = os.path.join(e4_folder, "HR.csv")
    _, hr_rate, hr_values = read_e4_csv(hr_path)

    if not hr_values:
        print(f"    HR.csv missing or empty — skipping")
        continue

    print(f"    HR.csv   : {len(hr_values)} readings")

    # ── Read IBI.csv → compute HRV ───────────────────────────
    ibi_path  = os.path.join(e4_folder, "IBI.csv")
    hrv_list  = compute_hrv_from_ibi(ibi_path)
    print(f"    IBI.csv  : {len(hrv_list)} HRV windows")

    # ── Read EDA.csv ─────────────────────────────────────────
    eda_path = os.path.join(e4_folder, "EDA.csv")
    _, eda_rate, eda_values = read_e4_csv(eda_path)

    # ── Read TEMP.csv ─────────────────────────────────────────
    temp_path = os.path.join(e4_folder, "TEMP.csv")
    _, temp_rate, temp_values = read_e4_csv(temp_path)

    # ── Build records ─────────────────────────────────────────
    # HR.csv is at 1 Hz — so index = second number
    # Take every 5th second to keep file size manageable
    min_len = len(hr_values)
    count   = 0

    for i in range(0, min_len, 5):

        # HR value
        hr = round(float(hr_values[i]), 1)
        hr = max(45, min(150, hr))

        # HRV — match index to available HRV windows
        if hrv_list:
            hrv_idx = min(i // 30, len(hrv_list) - 1)
            hrv     = round(float(hrv_list[hrv_idx]), 1)
        else:
            # Estimate HRV from HR if IBI not available
            hrv = round(max(8, min(80, 90 - hr * 0.5 + np.random.normal(0, 3))), 1)

        hrv = max(5, min(100, hrv))

        # EDA value
        eda = 0.0
        if eda_values and eda_rate:
            eda_idx = min(int(i * eda_rate), len(eda_values) - 1)
            eda     = round(float(eda_values[eda_idx]), 4)

        # Wrist temperature
        wrist_temp = 33.0
        if temp_values and temp_rate:
            temp_idx   = min(int(i * temp_rate), len(temp_values) - 1)
            wrist_temp = round(float(temp_values[temp_idx]), 2)

        # Infer label
        label = infer_label(hr, hrv)

        # SpO2 — derived from label and small noise
        # Stress → slightly lower SpO2
        spo2 = round(
            98.5 - (label == 2) * 1.2 + np.random.normal(0, 0.15),
            1
        )
        spo2 = max(93.0, min(100.0, spo2))

        all_records.append({
            'subject'    : sid,
            'hr'         : hr,
            'hrv'        : hrv,
            'eda'        : eda,
            'wrist_temp' : wrist_temp,
            'spo2'       : spo2,
            'label'      : label
        })
        count += 1

    print(f"    Records  : {count} rows saved from S{sid}")


# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────

if not all_records:
    print("\n❌ No records extracted.")
    print("   Check that WESAD_FOLDER path is correct.")
    print(f"   Current path: {WESAD_FOLDER}")

else:
    df = pd.DataFrame(all_records)

    # Make sure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Save
    df.to_csv(OUTPUT_CSV, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("  CONVERSION COMPLETE")
    print("=" * 60)
    print(f"  Total rows     : {len(df)}")
    print(f"  Subjects done  : {df['subject'].nunique()}")
    print(f"  HR range       : {df['hr'].min()} – {df['hr'].max()} bpm")
    print(f"  HRV range      : {df['hrv'].min()} – {df['hrv'].max()} ms")
    print(f"  SpO2 range     : {df['spo2'].min()} – {df['spo2'].max()} %")
    print(f"\n  Label breakdown:")

    names = {1: 'Baseline  (Normal/Focused)',
             2: 'Stress    (Stressed)',
             3: 'Amusement (Relaxed)'}

    for lbl, name in names.items():
        count = len(df[df['label'] == lbl])
        pct   = round(count / len(df) * 100, 1)
        print(f"    Label {lbl} — {name} : {count} rows ({pct}%)")

    print(f"\n  ✅ File saved to:")
    print(f"     {OUTPUT_CSV}")
    print("\n  Open File Explorer → data/ folder to verify.")
    print("=" * 60)