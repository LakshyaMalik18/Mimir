# Databricks notebook source
# MAGIC %md
# MAGIC # Facility Trust Desk — deterministic scorer
# MAGIC For each facility × claimed capability, compute a three-way verdict:
# MAGIC **Corroborated** (≥2 independent fields support), **Unverified** (claim present, nothing supports — missing data ≠ false),
# MAGIC **Contradicted** (a relevant field is PRESENT and conflicts).
# MAGIC Core rule: blank fields push toward Unverified, never Contradicted.

# COMMAND ----------

# UPDATE THIS to your exact table path (right-click table in Catalog -> copy path)
SRC = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.facilities"
OUT = "workspace.default.facility_verdicts"   # results table we create

# COMMAND ----------

# ---- Cell 1: load + inspect raw field formats (run this, eyeball output once) ----
import pandas as pd, re, json

df = spark.table(SRC).select(
    "unique_id","name","address_city","address_stateOrRegion",
    "description","capability","procedure","equipment","specialties",
    "numberDoctors","capacity"
).toPandas()

print(len(df), "rows")
for col in ["capability","equipment","procedure","numberDoctors"]:
    print(f"\n--- {col} samples ---")
    print(df[col].dropna().head(5).to_list())

# COMMAND ----------

# ---- Cell 2: helpers + capability evidence profiles ----

def norm(s):
    """camelCase -> spaced words, then lowercase. 'criticalCareMedicine' -> 'critical care medicine'"""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(s)).lower().strip()

def parse_list(val):
    """Handle JSON arrays (double-quoted), python-repr lists, or delimited strings.
    Returns normalized lowercase list or None if blank."""
    if val is None: return None
    s = str(val).strip()
    if s == "" or s.lower() in ("none","null","nan","[]"): return None
    for candidate in (s, s.replace("'", '"')):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                out = [norm(x) for x in obj if str(x).strip()]
                return out or None
        except Exception:
            continue
    parts = re.split(r"[;,|]", s)
    out = [norm(p) for p in parts if p.strip()]
    return out or None

def parse_num(val):
    if val is None: return None
    m = re.search(r"\d+", str(val))
    return int(m.group()) if m else None

def txt(val):
    if val is None: return None
    s = str(val).strip()
    return s.lower() if s and s.lower() not in ("none","null","nan") else None

# Evidence profiles: terms that corroborate each capability, per field
PROFILES = {
    "icu": dict(
        claim=["icu","intensive care","critical care"],   # matches 'critical care medicine' after norm()
        desc=["intensive care","critical care","ventilator","icu"],
        equip=["ventilator","cardiac monitor","defibrillator","icu bed","multipara monitor","infusion pump"],
        proc=["mechanical ventilation","critical care","intensive care","intubation"],
    ),
    "maternity": dict(
        claim=["maternity","obstetric","gynec","labor","delivery"],
        desc=["maternity","obstetric","labor","delivery","gynec","antenatal"],
        equip=["fetal monitor","incubator","delivery bed","ctg","ultrasound"],
        proc=["c-section","cesarean","caesarean","normal delivery","delivery","obstetric"],
    ),
    "emergency": dict(
        claim=["emergency","casualty","trauma center","24x7","ambulance"],
        desc=["emergency","casualty","24x7","24/7","round the clock","ambulance"],
        equip=["defibrillator","ambulance","crash cart","oxygen","stretcher"],
        proc=["emergency","resuscitation","first aid","triage"],
    ),
    "oncology": dict(
        claim=["oncology","cancer","tumor","tumour"],
        desc=["oncology","cancer","chemotherapy","radiation","tumor","tumour"],
        equip=["linear accelerator","chemotherapy","radiotherapy","brachytherapy","pet scan","ct scan"],
        proc=["chemotherapy","radiation therapy","radiotherapy","oncology","tumor removal","mastectomy"],
    ),
    "trauma": dict(
        claim=["trauma","accident","orthopedic emergency"],
        desc=["trauma","accident","fracture","orthopedic","casualty"],
        equip=["c-arm","x-ray","ct scan","orthopedic implants","stretcher","ventilator"],
        proc=["trauma","fracture","orthopedic surgery","amputation","wound"],
    ),
    "nicu": dict(
        claim=["nicu","neonatal","neonatology","newborn intensive"],
        desc=["nicu","neonatal","newborn","premature","preterm"],
        equip=["incubator","phototherapy","warmer","cpap","neonatal ventilator"],
        proc=["neonatal","newborn care","phototherapy","kangaroo"],
    ),
}

def any_term(terms, haystack):
    return any(t in haystack for t in terms)

def any_term_list(terms, items):
    joined = " | ".join(items)
    return any(t in joined for t in terms)

# COMMAND ----------

# ---- Cell 3: score every facility x claimed capability ----

rows = []
for _, r in df.iterrows():
    desc  = txt(r["description"])
    cap_l = parse_list(r["capability"])
    spec_l= parse_list(r["specialties"])
    proc_l= parse_list(r["procedure"])
    eq_l  = parse_list(r["equipment"])
    docs  = parse_num(r["numberDoctors"])
    beds  = parse_num(r["capacity"])

    claim_text = " | ".join((cap_l or []) + (spec_l or []))
    if not claim_text and desc is None:
        continue

    for cap_key, P in PROFILES.items():
        claimed = any_term(P["claim"], claim_text)
        if not claimed:
            continue

        # per-field: (present?, match?)
        sig = {
            "description": (desc is not None,  desc is not None and any_term(P["desc"], desc)),
            "equipment":   (eq_l is not None,  eq_l is not None and any_term_list(P["equip"], eq_l)),
            "procedure":   (proc_l is not None, proc_l is not None and any_term_list(P["proc"], proc_l)),
            "staffing":    (docs is not None,  docs is not None and docs >= 2),
        }

        support  = sum(1 for p, m in sig.values() if m)
        present  = sum(1 for p, m in sig.values() if p)

        # ---- CONTRADICTION: only a genuine prerequisite conflict, never "keyword absent" ----
        # Core rule: missing/unmatched data pushes to Unverified, NEVER Contradicted.
        # A contradiction requires a PRESENT field to explicitly conflict with a hard prerequisite.
        conflict_reasons = []

        # 1. Facility staffing is explicitly zero for a capability that needs doctors.
        if docs is not None and docs == 0:
            conflict_reasons.append("staffing data present but lists 0 doctors")

        # 2. Explicit negation of the capability in the description
        #    (e.g. "no ICU", "does not offer emergency", "icu not available").
        if desc is not None:
            for term in P["desc"]:
                if re.search(r"\b(no|not|without|lacks?|does not (have|offer)|unavailable)\b[\w\s]{0,25}" + re.escape(term), desc) \
                   or re.search(re.escape(term) + r"[\w\s]{0,15}\b(not available|unavailable|not offered)\b", desc):
                    conflict_reasons.append("description explicitly negates " + cap_key)
                    break

        if conflict_reasons:
            verdict = "Contradicted"
        elif support >= 2:
            verdict = "Corroborated"
        else:
            verdict = "Unverified"

        rows.append(dict(
            unique_id=r["unique_id"], name=r["name"],
            city=r["address_city"], state=r["address_stateOrRegion"],
            capability=cap_key, verdict=verdict,
            evidence_tier=support,
            data_completeness=round(present/4, 2),
            desc_match=sig["description"][1], equip_match=sig["equipment"][1],
            proc_match=sig["procedure"][1], staff_match=sig["staffing"][1],
            desc_present=sig["description"][0], equip_present=sig["equipment"][0],
            proc_present=sig["procedure"][0], staff_present=sig["staffing"][0],
            conflict_reason="; ".join(conflict_reasons) if conflict_reasons else None,
            num_doctors=docs, capacity=beds,
        ))

verdicts = pd.DataFrame(rows)
print(len(verdicts), "facility-capability claims scored")
print(verdicts["verdict"].value_counts())
print(verdicts.groupby("capability")["verdict"].value_counts().head(20))

# COMMAND ----------

# ---- Cell 4: sanity-check the story, then persist ----
# Amber vs red must be meaningfully different populations:
print(verdicts[verdicts.verdict=="Unverified"]["data_completeness"].describe())
print(verdicts[verdicts.verdict=="Contradicted"]["data_completeness"].describe())

spark.createDataFrame(verdicts).write.mode("overwrite").saveAsTable(OUT)
print("saved ->", OUT)
