"""Gauge, 'value vs typical range' charts and lifestyle / next-step advice."""
import math

import pandas as pd

from ml.config import DATASETS


def R(field, label, unit, lo, hi, amin, amax):
    return dict(field=field, label=label, unit=unit, lo=lo, hi=hi, amin=amin, amax=amax)


REFERENCE = {
    "heart": [
        R("trestbps", "Resting blood pressure", "mm Hg", 90, 130, 60, 200),
        R("chol", "Serum cholesterol", "mg/dl", 125, 200, 100, 350),
        R("oldpeak", "ST depression", "mm", 0, 1, 0, 6),
    ],
    "kidney": [
        R("sc", "Serum creatinine", "mg/dl", 0.6, 1.3, 0, 8),
        R("hemo", "Haemoglobin", "g/dl", 12, 17.5, 5, 18),
        R("bu", "Blood urea", "mg/dl", 7, 20, 0, 150),
        R("bgr", "Random blood glucose", "mg/dl", 70, 140, 50, 400),
        R("bp", "Blood pressure (diastolic)", "mm Hg", 60, 80, 40, 140),
        R("sod", "Sodium", "mEq/L", 135, 145, 120, 160),
        R("pot", "Potassium", "mEq/L", 3.5, 5.0, 2, 7),
        R("pcv", "Packed cell volume", "%", 36, 50, 15, 55),
        R("rbcc", "Red blood cell count", "millions/cmm", 4.2, 6.1, 2, 8),
        R("wbcc", "White blood cell count", "cells/cumm", 4000, 11000, 2000, 20000),
    ],
}
REF_NOTE = {
    "heart": "Green band = usual target range (blood pressure target is for people with diabetes).",
    "kidney": "Green band = usual laboratory reference range. Your lab's own ranges may differ slightly.",
    "eye": "Green band = middle 50% of training patients WITHOUT retinopathy. These image measures have no clinical normal range.",
}

URGENT = {
    "heart": "Seek emergency care straight away for chest pain or pressure, pain spreading to the arm or jaw, sudden breathlessness or fainting.",
    "kidney": "Seek urgent care for marked swelling, very little urine, severe breathlessness, confusion or persistent vomiting.",
    "eye": "Get urgent eye care for sudden vision loss, a shower of new floaters, flashes of light, or a dark curtain across your vision.",
}
NEXT = {
    "heart": {
        "low": ["Keep your routine check-ups: blood pressure, cholesterol (lipid profile) and HbA1c at least once a year.",
                "Ask your doctor how often your heart risk should be reviewed."],
        "moderate": ["Book a visit with your doctor in the next few weeks to review your heart risk.",
                     "Ask about a lipid profile, a resting ECG and whether more heart tests are needed.",
                     "Bring your recent HbA1c and blood-pressure readings."],
        "high": ["See your doctor or a cardiologist soon; do not wait for your next routine visit.",
                 "Ask about an ECG, a stress test, and whether blood-pressure or cholesterol treatment should be started or adjusted."]},
    "kidney": {
        "low": ["People with diabetes should have a urine albumin-to-creatinine ratio (uACR) and a blood creatinine / eGFR test at least once a year."],
        "moderate": ["Ask your doctor for uACR and eGFR tests soon, and repeat them to see whether the result persists.",
                     "Review your medicines with your doctor or pharmacist for kidney safety."],
        "high": ["See your doctor or a kidney specialist (nephrologist) promptly for confirmatory blood and urine tests.",
                 "Do not start or stop any medicine on your own; ask which ones may need adjusting."]},
    "eye": {
        "low": ["Everyone with diabetes needs a dilated eye examination at least once a year, even with good vision."],
        "moderate": ["Book an eye examination with an ophthalmologist within the next few weeks.",
                     "Tell them how long you have had diabetes and your recent HbA1c."],
        "high": ["See an ophthalmologist soon for a dilated exam or retinal imaging; early treatment prevents most vision loss."]},
}
LIFESTYLE = {
    "heart": ["Aim for about 150 minutes a week of moderate activity such as brisk walking, unless your doctor advises otherwise.",
              "Favour vegetables, whole grains, pulses, nuts, fish and olive oil; limit fried food, sweets and processed meat.",
              "Keep salt low (under about 5 g a day) to help blood pressure.",
              "Do not smoke and keep alcohol low.",
              "Keep blood sugar near the target your doctor set; steady control protects blood vessels.",
              "Take prescribed medicines regularly and check blood pressure at home if advised."],
    "kidney": ["Keep blood pressure and blood sugar within the targets your doctor sets: they are the biggest protectors of the kidneys.",
               "Cut down on salt and processed food.",
               "Avoid regular use of anti-inflammatory painkillers (ibuprofen, diclofenac) unless your doctor agrees.",
               "Drink water normally unless you were told to restrict fluids.",
               "Do not smoke, stay physically active and keep a healthy weight.",
               "Ask your doctor what amount of protein suits you rather than cutting it drastically."],
    "eye": ["Keep blood sugar, blood pressure and cholesterol within your doctor's targets; this slows retinopathy.",
            "Do not skip screening because your vision seems fine: early damage has no symptoms.",
            "Do not smoke.",
            "Report blurred vision, floaters or dark spots to your doctor early."],
}
TIPS = {
    ("trestbps", "high"): "Your resting blood pressure is above the usual target for people with diabetes (about 130/80). Recheck it at rest on different days and discuss it with your doctor.",
    ("chol", "high"): "Your cholesterol is above 200 mg/dl. Ask whether diet changes or cholesterol-lowering medicine are right for you.",
    ("oldpeak", "high"): "ST depression above 1 mm during exercise can indicate reduced blood flow to the heart and deserves a doctor's review.",
    ("sc", "high"): "Serum creatinine above the usual range can mean reduced kidney filtering; ask for an eGFR calculation.",
    ("bp", "high"): "Your blood pressure is above the usual range; tight control protects the kidneys in diabetes.",
    ("bgr", "high"): "Your blood glucose reading is above the typical range. Follow your diabetes plan and tell your doctor if readings stay high.",
    ("hemo", "low"): "Low haemoglobin (anaemia) is common in kidney disease; ask your doctor about it.",
    ("pot", "high"): "High potassium can affect the heart. Do not take potassium supplements or salt substitutes without asking your doctor.",
}
EYE_TIP = "Some retinal measurements are above the typical range for patients without retinopathy; an eye specialist can interpret what they mean for you."


def gauge(p):
    cx, cy, r = 100, 100, 80

    def pt(v, rad=r):
        th = math.pi * (1 - v / 100)
        return cx + rad * math.cos(th), cy - rad * math.sin(th)

    segs = []
    for a, b, col in ((0, 30, "var(--good)"), (30, 60, "var(--amber)"), (60, 100, "var(--coral)")):
        (x0, y0), (x1, y1) = pt(a), pt(b)
        segs.append(dict(d=f"M{x0:.1f} {y0:.1f} A{r} {r} 0 0 1 {x1:.1f} {y1:.1f}", color=col))
    nx, ny = pt(p, 62)
    return dict(segs=segs, nx=round(nx, 1), ny=round(ny, 1))


def _items(key, report):
    if key != "eye":
        return REFERENCE[key]
    labels = {f["name"]: f["label"] for f in DATASETS["eye"]["fields"]}
    out = []
    for f, (lo, hi, amin, amax) in report.get("healthy_ranges", {}).items():
        amax = max(amax, hi * 1.05, amin + 1e-6)
        out.append(R(f, labels[f], "", lo, hi, min(amin, lo), amax))
    return out


def charts(key, df, report):
    rows = []
    for it in _items(key, report):
        v = df.iloc[0].get(it["field"])
        if v is None or pd.isna(v):
            continue
        v, a0, a1 = float(v), it["amin"], it["amax"]
        span = a1 - a0

        def pct(x):
            return round(min(max((x - a0) / span * 100, 0), 100), 1)
        status = "low" if v < it["lo"] else "high" if v > it["hi"] else "normal"
        rows.append(dict(field=it["field"], label=it["label"], unit=it["unit"], value=f"{v:g}", status=status,
                         pos=pct(v), left=pct(it["lo"]), width=round(pct(it["hi"]) - pct(it["lo"]), 1),
                         lo=f"{it['lo']:.4g}", hi=f"{it['hi']:.4g}"))
    return rows


def advice(key, band, rows):
    tips = [TIPS[(r["field"], r["status"])] for r in rows if (r["field"], r["status"]) in TIPS]
    if key == "eye" and any(r["status"] == "high" for r in rows):
        tips.append(EYE_TIP)
    return dict(next=NEXT[key][band], lifestyle=LIFESTYLE[key], tips=tips, urgent=URGENT[key],
                show_urgent=band != "low")


def build_result(key, df, prob, report):
    p = float(prob)
    band, band_label = (("low", "Low risk") if p < 0.30 else ("moderate", "Moderate risk") if p < 0.60
                        else ("high", "High risk"))
    cfg, rows = DATASETS[key], None
    rows = charts(key, df, report)
    return dict(probability=round(p * 100, 1), band=band, band_label=band_label,
                label={"low": cfg["negative"], "moderate": cfg["possible"], "high": cfg["positive"]}[band], gauge=gauge(p * 100),
                charts=rows, chart_note=REF_NOTE[key], advice=advice(key, band, rows))
