"""Dataset definitions + web-form field specs (single source of truth)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS, REPORTS = ROOT / "data", ROOT / "models", ROOT / "reports"
PLOTS = ROOT / "static" / "plots"


def num(name, label, unit="", lo=None, hi=None, step="any", req=False):
    return dict(name=name, label=label, kind="num", unit=unit, min=lo, max=hi,
                step=step, required=req)


def sel(name, label, options, req=False, dtype="cat"):
    return dict(name=name, label=label, kind="select", options=options,
                required=req, dtype=dtype)


YN = [("yes", "Yes"), ("no", "No")]
NORMAL = [("normal", "Normal"), ("abnormal", "Abnormal")]
PRESENT = [("present", "Present"), ("notpresent", "Not present")]
SCALE = [(str(i), str(i)) for i in range(6)]

HEART_FIELDS = [
    num("age", "Age", "years", 1, 120, 1, True),
    sel("sex", "Sex", [("1", "Male"), ("0", "Female")], True),
    sel("cp", "Chest pain type", [("1", "Typical angina"), ("2", "Atypical angina"),
                                  ("3", "Non-anginal pain"), ("4", "Asymptomatic")], True),
    num("trestbps", "Resting blood pressure", "mm Hg", 60, 260, 1, True),
    num("chol", "Serum cholesterol", "mg/dl", 80, 700, 1, True),
    num("thalach", "Maximum heart rate achieved", "bpm", 50, 230, 1, True),
    num("oldpeak", "ST depression (exercise vs rest)", "mm", 0, 8, 0.1),
    sel("fbs", "Fasting blood sugar > 120 mg/dl", [("1", "Yes"), ("0", "No")]),
    sel("restecg", "Resting ECG", [("0", "Normal"), ("1", "ST-T wave abnormality"),
                                   ("2", "Left ventricular hypertrophy")]),
    sel("exang", "Exercise-induced angina", [("1", "Yes"), ("0", "No")]),
    sel("slope", "Slope of peak-exercise ST segment",
        [("1", "Upsloping"), ("2", "Flat"), ("3", "Downsloping")]),
    sel("ca", "Major vessels coloured by fluoroscopy", [(str(i), str(i)) for i in range(4)]),
    sel("thal", "Thalassemia (thallium scan)",
        [("3", "Normal"), ("6", "Fixed defect"), ("7", "Reversible defect")]),
]

KIDNEY_FIELDS = [
    num("age", "Age", "years", 1, 120, 1, True),
    num("bp", "Blood pressure (diastolic)", "mm Hg", 40, 200, 1, True),
    num("sc", "Serum creatinine", "mg/dl", 0.1, 80, 0.1, True),
    num("hemo", "Haemoglobin", "g/dl", 2, 22, 0.1, True),
    num("bu", "Blood urea", "mg/dl", 1, 400, 0.1),
    num("bgr", "Blood glucose (random)", "mg/dl", 20, 600, 1),
    num("sod", "Sodium", "mEq/L", 80, 170, 0.1),
    num("pot", "Potassium", "mEq/L", 1, 50, 0.1),
    num("pcv", "Packed cell volume", "%", 5, 70, 1),
    num("wbcc", "White blood cell count", "cells/cumm", 1000, 30000, 100),
    num("rbcc", "Red blood cell count", "millions/cmm", 1, 9, 0.1),
    sel("sg", "Urine specific gravity",
        [(v, v) for v in ("1.005", "1.010", "1.015", "1.020", "1.025")], dtype="num"),
    sel("al", "Urine albumin (0-5)", SCALE, dtype="num"),
    sel("su", "Urine sugar (0-5)", SCALE, dtype="num"),
    sel("rbc", "Red blood cells (urine)", NORMAL),
    sel("pc", "Pus cells", NORMAL),
    sel("pcc", "Pus cell clumps", PRESENT),
    sel("ba", "Bacteria", PRESENT),
    sel("htn", "Hypertension", YN),
    sel("dm", "Diabetes mellitus", YN),
    sel("cad", "Coronary artery disease", YN),
    sel("appet", "Appetite", [("good", "Good"), ("poor", "Poor")]),
    sel("pe", "Pedal edema", YN),
    sel("ane", "Anaemia", YN),
]


BIN = lambda a, b: [("1", a), ("0", b)]
EYE_FIELDS = [
    sel("quality", "Retinal image quality", BIN("Good", "Poor"), True, "num"),
    sel("prescreen", "Severe retinal abnormality at pre-screening", BIN("Yes", "No"), True, "num"),
] + [num(f"ma{c}", f"Microaneurysm count (confidence {v})", "count", 0, 300, 1, True)
     for c, v in zip(("05", "06", "07", "08", "09", "10"), ("0.5", "0.6", "0.7", "0.8", "0.9", "1.0"))
] + [num(f"ex{i}", f"Exudate measure {i}", "normalised area", 0, 500, "any", True) for i in range(1, 9)] + [
    num("macula_od", "Distance: macula centre to optic-disc centre", "normalised", 0.2, 0.8, "any", True),
    num("od_diam", "Optic disc diameter", "normalised", 0.02, 0.4, "any", True),
    sel("ammfm", "AM/FM-based classification", BIN("Signs of DR", "No signs"), True, "num"),
]
EYE_COLS = ["quality", "prescreen"] + [f"ma{c}" for c in ("05", "06", "07", "08", "09", "10")] + \
           [f"ex{i}" for i in range(1, 9)] + ["macula_od", "od_diam", "ammfm"]

HEART_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
              "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
KIDNEY_COLS = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot",
               "hemo", "pcv", "wbcc", "rbcc", "htn", "dm", "cad", "appet", "pe", "ane"]  # ARFF column order

DATASETS = {
    "eye": dict(title="Diabetic Retinopathy (Eye)", short="Eye", fields=EYE_FIELDS,
                positive="Signs of retinopathy likely", possible="Signs of retinopathy possible", negative="Signs of retinopathy unlikely",
                source="UCI Diabetic Retinopathy Debrecen (Messidor image features)",
                blurb="Predicts diabetic retinopathy from features extracted from retinal fundus images."),
    "heart": dict(title="Heart Disease", short="Heart", fields=HEART_FIELDS,
                  blurb="Predicts cardiovascular disease from blood pressure, cholesterol, ECG and exercise-test results.",
                  positive="Heart disease likely", possible="Heart disease possible", negative="Heart disease unlikely",
                  source="UCI Heart Disease: Cleveland, Hungarian, Switzerland, VA Long Beach"),
    "kidney": dict(title="Chronic Kidney Disease", short="Kidney", fields=KIDNEY_FIELDS,
                   blurb="Predicts chronic kidney disease from blood and urine tests and clinical history.",
                   positive="Kidney disease likely", possible="Kidney disease possible", negative="Kidney disease unlikely",
                   source="UCI Chronic Kidney Disease: Apollo Hospitals, Tamil Nadu"),
}
DATASETS = {k: DATASETS[k] for k in ("heart", "kidney", "eye")}   # display order
for d in DATASETS.values():
    d["num_cols"] = [f["name"] for f in d["fields"] if f["kind"] == "num" or f.get("dtype") == "num"]
    d["cat_cols"] = [f["name"] for f in d["fields"] if f["name"] not in d["num_cols"]]
