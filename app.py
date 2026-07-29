"""
ER Triage Kiosk — single-file Streamlit app
=============================================
Run with:
    pip install streamlit torch scikit-learn joblib pandas numpy wordninja
    streamlit run triage_app.py

Expects a model_artifacts/ folder next to this file (copy it from wherever
esi_nn_v2.py wrote it), containing:
    nn_search_best.pt, nn_search_best_config.json, scaler.joblib,
    feature_columns.json, label_classes.json

That's it — no separate backend, no mobile build step. Streamlit serves this
as a real web page in your browser, and the in-memory patient queue is shared
across every browser tab hitting this same running process (so a "kiosk" tab
and a "staff" tab both talk to the same queue).
"""
import itertools
import json
import os
import re
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import wordninja
from torch import nn

MODEL_DIR = "model_artifacts"
NURSE_ACCESS_CODE = os.environ.get("NURSE_ACCESS_CODE", "1234")

st.set_page_config(page_title="ER Triage Kiosk", page_icon="🏥", layout="centered")


# ---------------------------------------------------------
# Model architecture (must match esi_nn_v2.py exactly)
# ---------------------------------------------------------
class ConfigurableNet(nn.Module):
    def __init__(self, n_features, hidden_dims, dropout, ordinal, n_classes=5):
        super().__init__()
        layers_list = []
        in_dim = n_features
        for h in hidden_dims:
            layers_list += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.trunk = nn.Sequential(*layers_list)
        self.ordinal = ordinal
        self.n_classes = n_classes
        if ordinal:
            self.shared_out = nn.Linear(in_dim, 1, bias=False)
            self.bias_0 = nn.Parameter(torch.zeros(1))
            self.bias_decrements_raw = nn.Parameter(torch.zeros(n_classes - 2))
        else:
            self.out = nn.Linear(in_dim, n_classes)

    def get_ordinal_biases(self):
        decrements = torch.nn.functional.softplus(self.bias_decrements_raw)
        zero = torch.zeros(1, device=decrements.device)
        return self.bias_0 - torch.cat([zero, torch.cumsum(decrements, dim=0)])

    def forward(self, x):
        h = self.trunk(x)
        if self.ordinal:
            return self.shared_out(h) + self.get_ordinal_biases()
        return self.out(h)


def coral_probs_to_class_probs(cum_probs, n_classes=5):
    batch = cum_probs.shape[0]
    p = np.zeros((batch, n_classes))
    p[:, 0] = 1 - cum_probs[:, 0]
    for k in range(1, n_classes - 1):
        p[:, k] = cum_probs[:, k - 1] - cum_probs[:, k]
    p[:, n_classes - 1] = cum_probs[:, n_classes - 2]
    p = np.clip(p, 0, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p


# ---------------------------------------------------------
# Plain-language symptom labels + duplicate merging (same glossary as before)
# ---------------------------------------------------------
GLOSSARY = {
    "abdominaldistention": "Abdominal bloating or swelling",
    "abdominalpainpregnant": "Abdominal pain (pregnant)",
    "abnormallab": "Abnormal lab result follow-up",
    "alteredmentalstatus": "Confusion or altered mental state",
    "bodyfluidexposure": "Exposure to body fluids",
    "coldlikesymptoms": "Cold-like symptoms",
    "decreasedbloodsugar-symptomatic": "Low blood sugar (with symptoms)",
    "detoxevaluation": "Drug or alcohol detox evaluation",
    "drug/alcoholassessment": "Drug or alcohol assessment",
    "dysuria": "Painful urination",
    "earpain": "Ear pain",
    "elevatedbloodsugar-nosymptoms": "High blood sugar (no symptoms)",
    "elevatedbloodsugar-symptomatic": "High blood sugar (with symptoms)",
    "emesis": "Vomiting",
    "epigastricpain": "Upper stomach pain",
    "epistaxis": "Nosebleed",
    "exposuretostd": "Possible STD/STI exposure",
    "extremitylaceration": "Cut on arm or leg",
    "extremityweakness": "Weakness in arm or leg",
    "fall>65": "Fall (age 65+)",
    "feverimmunocompromised": "Fever with weakened immune system",
    "fever-75yearsorolder": "Fever (age 75+)",
    "fever-9weeksto74years": "Fever (9 weeks to 74 years old)",
    "follow-upcellulitis": "Skin infection follow-up",
    "foreignbodyineye": "Something in eye",
    "fulltrauma": "Major trauma or injury",
    "generalizedbodyaches": "Body aches",
    "giproblem": "Stomach or digestive problem",
    "headache-newonsetornewsymptoms": "New headache or new headache symptoms",
    "headache-recurrentorknowndxmigraines": "Recurring headache or known migraines",
    "headachere-evaluation": "Headache follow-up",
    "hematuria": "Blood in urine",
    "hemoptysis": "Coughing up blood",
    "hyperglycemia": "High blood sugar",
    "hypertension": "High blood pressure",
    "hypotension": "Low blood pressure",
    "ingestion": "Swallowed something harmful",
    "irregularheartbeat": "Irregular heartbeat",
    "lossofconsciousness": "Passed out or lost consciousness",
    "maleguproblem": "Male genital or urinary problem",
    "femaleguproblem": "Female genital or urinary problem",
    "medicalscreening": "General medical screening",
    "modifiedtrauma": "Trauma or injury evaluation",
    "motorcyclecrash": "Motorcycle accident",
    "motorvehiclecrash": "Car accident",
    "multiplefalls": "Multiple falls",
    "nearsyncope": "Almost fainted or lightheaded",
    "oralswelling": "Swelling in mouth",
    "otalgia": "Ear pain",
    "palpitations": "Racing or fluttering heartbeat",
    "post-opproblem": "Problem after surgery",
    "psychoticsymptoms": "Hearing or seeing things that aren't there",
    "rapidheartrate": "Rapid heart rate",
    "respiratorydistress": "Severe trouble breathing",
    "seizure-newonset": "First-time seizure",
    "seizure-priorhxof": "Seizure (history of seizures)",
    "sicklecellpain": "Sickle cell pain crisis",
    "stdcheck": "STD/STI test request",
    "suture/stapleremoval": "Stitch or staple removal",
    "swallowedforeignbody": "Swallowed an object",
    "syncope": "Fainted",
    "tachycardia": "Rapid heart rate",
    "tickremoval": "Tick removal",
    "uri": "Cold or upper respiratory symptoms",
    "urinaryretention": "Can't urinate",
    "withdrawal-alcohol": "Alcohol withdrawal symptoms",
    "woundre-evaluation": "Wound follow-up",
}
MANUAL_MERGE_GROUPS = [
    {"cc_arminjury", "cc_arminjury/pain"},
    {"cc_handinjury", "cc_handinjury/pain"},
]


def _auto_label(raw: str) -> str:
    s = re.sub(r"[-/>]", " ", raw)
    words = []
    for part in s.split():
        words.extend(wordninja.split(part))
    text = " ".join(words)
    return (text[:1].upper() + text[1:]) if text else text


def _cc_label(cc_col: str) -> str:
    raw = cc_col[len("cc_"):]
    if raw in GLOSSARY:
        return GLOSSARY[raw]
    if raw.endswith("injury/pain"):
        bodypart = " ".join(wordninja.split(raw[: -len("injury/pain")]))
        return f"{bodypart.capitalize()} injury or pain"
    return _auto_label(raw)


def build_symptom_groups(cc_columns):
    cc_to_label = {c: _cc_label(c) for c in cc_columns}
    assigned, groups = {}, {}
    for merge_set in MANUAL_MERGE_GROUPS:
        present = [c for c in merge_set if c in cc_to_label and c not in assigned]
        if len(present) > 1:
            group_key = f"group_{present[0]}"
            label = min((cc_to_label[c] for c in present), key=len)
            groups[group_key] = {"label": label, "columns": present}
            for c in present:
                assigned[c] = group_key
    by_label = {}
    for c in cc_columns:
        if c in assigned:
            continue
        by_label.setdefault(cc_to_label[c], []).append(c)
    for label, cols in by_label.items():
        group_key = f"group_{cols[0]}"
        groups[group_key] = {"label": label, "columns": cols}
        for c in cols:
            assigned[c] = group_key
    return groups, cc_to_label


# ---------------------------------------------------------
# Load model + artifacts once, shared across all sessions
# ---------------------------------------------------------
@st.cache_resource
def load_everything():
    with open(os.path.join(MODEL_DIR, "nn_search_best_config.json")) as f:
        config = json.load(f)
    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)
    with open(os.path.join(MODEL_DIR, "label_classes.json")) as f:
        label_classes = json.load(f)
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

    use_ordinal = config.get("use_ordinal", True)
    n_classes = config.get("n_classes", len(label_classes))
    override_cols = config.get("override_cols", [])
    cc_columns = [c for c in feature_columns if c.startswith("cc_")]

    model = ConfigurableNet(
        n_features=len(feature_columns),
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=config["dropout"],
        ordinal=use_ordinal,
        n_classes=n_classes,
    )
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "nn_search_best.pt"), map_location="cpu"))
    model.eval()

    feature_index = {name: i for i, name in enumerate(feature_columns)}
    symptom_groups, cc_to_label = build_symptom_groups(cc_columns)

    return {
        "config": config, "feature_columns": feature_columns, "label_classes": label_classes,
        "scaler": scaler, "model": model, "use_ordinal": use_ordinal, "n_classes": n_classes,
        "override_cols": override_cols, "cc_columns": cc_columns, "feature_index": feature_index,
        "symptom_groups": symptom_groups, "cc_to_label": cc_to_label,
    }


@st.cache_resource
def get_queue_store():
    return {"patients": {}, "counter": itertools.count(1)}


def run_prediction(art, age, gender, cc_cols):
    fi, fc = art["feature_index"], art["feature_columns"]
    row = np.zeros(len(fc), dtype="float32")
    row[fi["age"]] = age
    row[fi["gender"]] = 1.0 if gender == "Male" else 0.0
    for s in cc_cols:
        row[fi[s]] = 1.0
    if "symptom_count" in fi:
        row[fi["symptom_count"]] = float(len(cc_cols))
    if "is_infant" in fi:
        row[fi["is_infant"]] = 1.0 if age < 1 else 0.0
    if "is_pediatric" in fi:
        row[fi["is_pediatric"]] = 1.0 if age < 18 else 0.0
    if "is_elderly" in fi:
        row[fi["is_elderly"]] = 1.0 if age >= 65 else 0.0

    scaled = art["scaler"].transform(pd.DataFrame([row], columns=fc)).astype("float32")
    x = torch.tensor(scaled)
    with torch.no_grad():
        logits = art["model"](x)
        if art["use_ordinal"]:
            cum_probs = torch.sigmoid(logits).numpy()
            probs = coral_probs_to_class_probs(cum_probs, art["n_classes"])[0]
        else:
            probs = torch.softmax(logits, dim=1).numpy()[0]

    esi_values = np.array(art["label_classes"])
    severity_score = float(probs @ esi_values)
    model_pred_idx = int(np.argmax(probs))

    triggered = [c for c in art["override_cols"] if c in cc_cols]
    if triggered:
        return 1, 1.0, True
    return int(esi_values[model_pred_idx]), severity_score, False


def resequence(store):
    waiting = sorted((p for p in store["patients"].values() if p["status"] == "waiting"),
                      key=lambda p: p["queue_position"])
    for i, p in enumerate(waiting):
        p["queue_position"] = i


ESI_COLORS = {1: "#DC2626", 2: "#EA580C", 3: "#CA8A04", 4: "#16A34A", 5: "#2563EB"}

# ---------------------------------------------------------
# App
# ---------------------------------------------------------
art = load_everything()
store = get_queue_store()

page = st.sidebar.radio("View", ["Patient Check-In", "Staff Queue"])

if page == "Patient Check-In":
    st.title("🏥 Welcome — Check In")
    st.write("Please enter your information and symptoms below.")

    name = st.text_input("Full name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)

    label_to_key = {v["label"]: k for k, v in art["symptom_groups"].items()}
    all_labels = sorted(label_to_key.keys())
    selected_labels = st.multiselect("Symptoms (select all that apply)", all_labels)

    if st.button("Submit Check-In", type="primary"):
        if not name.strip():
            st.error("Please enter your name.")
        elif not selected_labels:
            st.error("Please select at least one symptom.")
        else:
            group_keys = [label_to_key[l] for l in selected_labels]
            cc_cols = set()
            for g in group_keys:
                cc_cols.update(art["symptom_groups"][g]["columns"])
            esi, severity, override_applied = run_prediction(art, float(age), gender, list(cc_cols))

            patient_id = next(store["counter"])
            waiting = [p for p in store["patients"].values() if p["status"] == "waiting"]
            insert_pos = sum(1 for p in waiting if p["severity_score"] <= severity)
            for p in waiting:
                if p["queue_position"] >= insert_pos:
                    p["queue_position"] += 1
            store["patients"][patient_id] = {
                "id": patient_id, "name": name.strip(), "age": age, "gender": gender,
                "symptoms": selected_labels, "esi": esi, "severity_score": severity,
                "override_applied": override_applied, "checked_in_at": time.time(),
                "status": "waiting", "queue_position": insert_pos,
            }

            st.success("✅ You're checked in!")
            if override_applied:
                st.warning("Please let a staff member know right away that you've checked in.")
            st.info("Please have a seat. A staff member will call you based on medical need, "
                    "which may not be the same order as check-in. Thank you for your patience.")

else:
    st.title("👩‍⚕️ Staff Queue")
    code = st.text_input("Access code", type="password")
    if code != NURSE_ACCESS_CODE:
        if code:
            st.error("Incorrect code.")
        st.stop()

    tab_waiting, tab_history = st.tabs(["Waiting", "Past Patients"])
    waiting = sorted((p for p in store["patients"].values() if p["status"] == "waiting"),
                      key=lambda p: p["queue_position"])
    history = sorted((p for p in store["patients"].values() if p["status"] == "seen"),
                      key=lambda p: p["checked_in_at"], reverse=True)

    with tab_waiting:
        st.write(f"**{len(waiting)} waiting**")
        for i, p in enumerate(waiting):
            cols = st.columns([1, 4, 1, 1, 1])
            cols[0].markdown(
                f"<div style='background:{ESI_COLORS[p['esi']]};color:white;border-radius:50%;"
                f"width:32px;height:32px;text-align:center;line-height:32px;font-weight:bold'>"
                f"{p['esi']}</div>", unsafe_allow_html=True)
            tag = " 🚩OVERRIDE" if p["override_applied"] else ""
            cols[1].write(f"**{p['name']}** — {p['age']}y {p['gender']} — "
                           f"severity {p['severity_score']:.2f}{tag}\n\n"
                           f"{', '.join(p['symptoms'])}")
            if cols[2].button("↑", key=f"up_{p['id']}", disabled=(i == 0)):
                other = waiting[i - 1]
                p["queue_position"], other["queue_position"] = other["queue_position"], p["queue_position"]
                resequence(store)
                st.rerun()
            if cols[3].button("↓", key=f"down_{p['id']}", disabled=(i == len(waiting) - 1)):
                other = waiting[i + 1]
                p["queue_position"], other["queue_position"] = other["queue_position"], p["queue_position"]
                resequence(store)
                st.rerun()
            if cols[4].button("Seen", key=f"seen_{p['id']}"):
                p["status"] = "seen"
                resequence(store)
                st.rerun()
        if not waiting:
            st.write("No patients waiting.")

    with tab_history:
        st.write(f"**{len(history)} seen**")
        for p in history:
            st.write(f"{p['name']} — {p['age']}y {p['gender']} — ESI {p['esi']} — "
                     f"{', '.join(p['symptoms'])}")
        if not history:
            st.write("No past patients yet.")