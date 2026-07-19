# MIMIR — Intelligence You Can Trust

**Live app (public, no login):** https://projectmimir.streamlit.app
**Native Databricks Apps deployment:** https://facility-trust-desk-7474652289430090.aws.databricksapps.com *(SSO-gated — Databricks Apps do not support anonymous access; the public Streamlit deployment above runs the identical codebase against the same Databricks backend)*
**Hack-Nation Global AI Hackathon · Challenge 4: Data Legend — Building the Trust Layer for Indian Healthcare · Team HN-4651 (solo)**

---

> **77% of the capability claims in India's national health-facility dataset cannot be verified against the facilities' own records. Planners route patients on them anyway.**
>
> **Mimir is that missing trust layer — delivered today as a planner tool, powered underneath by a deterministic verification pipeline.** The layer is the vision. The tool is what you can click right now. The pipeline is how it works.

## Why "Mimir"

In Norse mythology, **Mimir** is the guardian of the well of wisdom — the being even Odin consults before making decisions that matter. Two things define him: his knowledge is **trusted because it comes from a verifiable source** (the well itself), and he **advises but never rules** — the decision always stays with the one who asks.

That is this system's exact contract. Mimir guards the well of facility data, tells you precisely what the evidence supports, cites the source for every answer — and leaves the final decision with the human planner. The eclipse in the logo is the well: a dark surface with light around its rim, because what Mimir shows you is not the whole truth — it is exactly how much of the truth the evidence can back, with the boundary drawn honestly.

An oracle asserts. Mimir cites. **Intelligence you can trust.**

---

## The problem, in plain English

An aid worker has a patient who needs an ICU bed tonight. She has a directory of 10,088 Indian health facilities. Every entry is **self-reported** — each hospital wrote its own description of what it can do. Nobody checked.

If she trusts a false claim, a patient is sent somewhere that can't treat them. If she distrusts everything, she abandons real facilities that just have thin paperwork. She has no way to tell the difference — and that difference is the whole problem. **A data desert is not a medical desert.** A facility we can't verify is not the same as a facility that's lying, and any tool that conflates the two hurts exactly the under-documented rural facilities that aid work exists for.

## What Mimir does

Mimir reads each facility's own records and asks one question per capability claim: **does the rest of your file back this up?** It answers with one of exactly three verdicts:

| Verdict | Meaning | Treatment |
|---|---|---|
| 🟢 **Corroborated** | The facility's other fields support the claim | Believable |
| 🔴 **Contradicted** | The facility's own records conflict with each other | Surfaced **first** — review before relying |
| 🟡 **Unverified** | Not enough data to say either way | **Reported, never punished** |

The amber rule is the soul of the project: *missing data is never treated as a "no."* A three-way verdict can honestly say "we don't know" — something a trust-score-out-of-100 structurally cannot.

## Why this is different

The default architecture for this kind of tool is: question → LLM → answer. The AI decides, and nobody can audit why.

Mimir splits the job in two, deliberately:

1. **A deterministic scorer decides trust.** Rules-based PySpark, no AI anywhere in the judgment. Run it a thousand times, same verdicts. Every rule is readable.
2. **The LLM only fetches proof.** On demand, it extracts and quotes the exact source sentence supporting a claim. It cites; it never judges.

**The AI shows its work — it doesn't do the deciding.** For a domain where a wrong answer costs a patient rather than a click, that is the architecture that's actually deployable.

Ranking is equally transparent, printed in the UI itself ("How this ranking works — no black box"): contradictions first, then evidence strength (0–4 fields), then data completeness as a tiebreaker. No composite magic number.

## Product walkthrough

1. **Pick a capability and region** (or describe the care you need in plain language — semantic search via Mosaic AI Vector Search; retrieval only, verdicts always come from the scorer).
2. **See the three-way verdict counts** and the ranked facility list, contradictions pinned to the top.
3. **Expand any facility** → a per-field evidence table (description / equipment / procedure / staffing), each marked *supports the claim*, *no data (not counted against)*, or *present, no match found* — with the raw evidence beside it.
4. **Conflict callouts in plain language** — e.g. *"Conflict: staffing data present but lists 0 doctors"* on a facility claiming critical care.
5. **Extract cited evidence (AI)** — one click; Llama 3.3 70B on Databricks Foundation Model serving quotes the exact source lines, cached to Delta.
6. **Planner override** — a human who has called the facility can overrule any verdict with a note, persisted for the next reviewer. The machine keeps the memory; the human keeps the authority.
7. **Trust map** — every facility plotted by verdict across India.

## Architecture

```
┌─────────────────────────────┐        ┌──────────────────────────────────┐
│  01_trust_scorer notebook   │        │            Mimir app             │
│  (offline, batch, PySpark)  │        │   (Databricks Apps + public      │
│                             │        │    Streamlit mirror, one repo)   │
│  10,088 facilities          │        │                                  │
│  → deterministic            │ writes │  reads verdicts live             │
│    cross-field checks       │───────►│  ├─ FM serving → cited receipts  │
│  → 18,100 verdicts          │ Delta  │  ├─ Vector Search → semantic     │
│  → MLflow run logging       │ table  │  ├─ Lakebase → overrides         │
│    (params + calibration)   │        │  │   (Delta fallback)            │
└─────────────────────────────┘        │  └─ pydeck trust map             │
                                       └──────────────────────────────────┘
```

The notebook and the app share **no code**. They meet at exactly one Delta table (`facility_verdicts`). Scoring is an offline, reproducible batch job; the app is a live reader. Changing the scorer means re-running the notebook — a deliberate audit boundary.

## Databricks stack — each service, and the job it does

| Service | Job |
|---|---|
| **Delta Lake** | Source of truth: 10,088 facilities, 18,100 verdicts, receipt cache, override fallback |
| **PySpark (notebook)** | Deterministic cross-field trust scorer over the full dataset |
| **Foundation Model serving** (Llama 3.3 70B) | Citation extraction only — quotes source text on demand, cached to Delta |
| **Mosaic AI Vector Search** | Semantic retrieval (9,997-row index, `databricks-gte-large-en` embeddings, endpoint `trust_desk_vs`) |
| **Lakebase (Postgres)** | Transactional persistence for planner overrides, with Delta fallback kill-switch |
| **MLflow** | Run logging for every scorer execution: parameters, verdict counts, calibration metrics, verdicts artifact |
| **Databricks Apps** | Native deployment (service-principal ambient auth) |

**Dual deployment from one codebase:** the app detects its environment (`DATABRICKS_CLIENT_ID` present → ambient service-principal auth; absent → PAT from secrets) and runs identically on Databricks Apps and on public Streamlit Cloud. Outside Databricks, overrides fall back to Delta (the in-app OAuth-token path is Databricks-only) — the toast reports the actual store used.

## The self-correction story (why the numbers can be trusted)

The first version of the contradiction rule flagged **5,113** facilities as Contradicted. Inspection showed it was largely punishing *missing* data — exactly the failure mode the product exists to avoid. The rule was tightened to require genuine internal conflict: **28** facilities.

Calibration confirms the fix is real, not cosmetic: **Contradicted facilities average 93% data completeness vs 77% for Unverified.** Red flags fire on rich records that disagree with themselves — not on thin records that are merely incomplete. Every scorer run, including this correction, is logged in MLflow with its parameters and metrics.

## Key numbers

- **10,088** facilities → **18,100** capability claims scored
- **5,113 → 28** contradiction false-positive correction
- **93% vs 77%** completeness calibration (Contradicted vs Unverified)
- **9,997** rows in the semantic search index
- **0** LLM calls in the trust decision path

## Repository

```
app.py                  # Mimir app (Databricks Apps + Streamlit Cloud, env-aware auth)
01_trust_scorer.py      # Deterministic scorer notebook (writes facility_verdicts, logs to MLflow)
requirements.txt
app.yaml.example        # Sanitized Databricks Apps config
secrets.toml.example    # Placeholder secrets for external deployment
manifest.yaml
```

## Running it

**Easiest:** open https://projectmimir.streamlit.app — no login.

**Reproduce the backend:** import `01_trust_scorer.py` into a Databricks workspace with the Virtue Foundation dataset shared in, run all cells (writes `workspace.default.facility_verdicts`, logs the MLflow run), then deploy `app.py` as a Databricks App or point a Streamlit deployment at the workspace using `secrets.toml.example`.

---

*Built solo in 24 hours. The thesis in one line: **an AI that asserts trust is a demo; a system that earns it is a product.***