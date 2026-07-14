import os
import csv
import numpy as np
from copy import deepcopy

from calculate_eer import compute_eer

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_PATH = os.path.join(BASE_DIR, "ASVspoof2019.LA.cm.eval.trl.txt")
SOLUTIONS_DIR = os.path.join(BASE_DIR, "students_solutions")
OUTPUT_CSV = os.path.join(BASE_DIR, "grades.csv")

# --- Load protocol ---
index = []
with open(PROTOCOL_PATH, "r") as protocol:
    for line in protocol:
        _, key, _, alg_id, label = line.strip().split()
        index.append({
            "key": key,
            "label": 1 if label == "bonafide" else 0
        })

# --- Prepare output ---
results = []

# --- Process each student file ---
for filename in os.listdir(SOLUTIONS_DIR):
    if filename.endswith(".csv"):
        name = filename.replace(".csv", "")
        filepath = os.path.join(SOLUTIONS_DIR, filename)

        # Load student scores into a dict
        try:
            student_scores = {}
            with open(filepath, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 2:
                        continue  # skip malformed lines
                    key, score = row
                    student_scores[key] = float(score)
        except Exception as e:
            print(filepath)
            raise e

        # Build student index
        student_index = deepcopy(index)
        for entry in student_index:
            key = entry["key"]
            entry["score"] = student_scores[key]  # error if missing

        # Extract scores and labels
        scores = np.array([entry["score"] for entry in student_index])
        labels = np.array([entry["label"] for entry in student_index])

        assert len(scores) == len(index), "Not enough / too many scores"

        # Compute EER
        bona_cm = scores[labels == 1]
        spoof_cm = scores[labels == 0]
        eer, _ = compute_eer(bona_cm, spoof_cm)

        eer *= 100 # in %

        # Grade calculation
        if eer > 10.9:
            grade = 0
        elif eer < 5.3:
            grade = 10
        else:
            # Linear interpolation between 2 and 10
            grade = 2 + (10.9 - eer) * (8 / (10.9 - 5.3))

        results.append({
            "name": name,
            "email": name + "@edu.hse.ru",
            "eer": round(eer, 4),
            "grade": round(grade, 2)
        })

# --- Write output CSV ---
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "email", "eer", "grade"])
    writer.writeheader()
    writer.writerows(results)

print(f"Grading complete. Results saved to {OUTPUT_CSV}")
