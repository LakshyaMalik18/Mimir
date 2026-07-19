# Mimir — Intelligence you can trust

**Live app:** https://facility-trust-desk-7474652289430090.aws.databricksapps.com
**Track:** Facility Trust Desk · Hack-Nation 6th Global AI Hackathon · Challenge 04 (Databricks "Data Legend")
**Built solo on Databricks Free Edition.**

> In Norse myth, Mimir guards the well of wisdom — Odin gave an eye for one drink from it.
> This Mimir guards a simpler promise: **every verdict cites its evidence.**

An NGO planner asks: *"Can this hospital actually do what it claims?"* Mimir answers with a verdict, the exact evidence behind it, and an honest account of what we don't know — across **10,088 Indian healthcare facilities and 18,100 capability claims.**

---

## The core idea: three verdicts, not a score

There is no ground truth in this dataset. A single 0–100 "trust score" would be false precision. Mimir instead classifies every capability claim into one of three auditable states:

| Verdict | Meaning | Count |
|---|---|---|
| 🟢 **Corroborated** | The facility's own records support the claim in independent fields | 4,135 |
| 🟡 **Unverified** | No evidence either way — a **data desert, not a medical desert** | 13,937 |
| 🔴 **Contradicted** | The facility's own data conflicts with the claim (e.g. claims ICU, lists 0 doctors) | 28 |

**The rule that matters: missing data ≠ contradiction.** Only 36% of records list doctor counts and 25% list capacity. A facility is never punished for a blank field — blanks lower *certainty*, never the *verdict*. The UI reports data completeness on every facility so the planner sees exactly what "we don't know" means. This is the challenge brief's data-desert problem, implemented as the product's core mechanic rather than a caveat.

## Self-correction: 5,113 false alarms → 28 real conflicts

Our first contradiction rule flagged **5,113** facilities red. Manual inspection showed most were data-formatting artifacts, not real conflicts — a trust tool that cries wolf 5,000 times is worse than no tool. We rebuilt the rule as a validator pass with internal-consistency checks (e.g. staffing data must be *present and zero* to conflict with a staffing-dependent claim, not merely absent). Result: **28 high-precision contradictions**, and the calibration holds up — Contradicted facilities average **93% data completeness** vs 77% for Unverified, meaning red flags fire where we have the *most* information, not the least. The scorer double-checks its own work; the numbers above are logged as MLflow artifacts for audit.

## Architecture: deterministic core, AI at the edges

```
                    OFFLINE (runs once)                      LIVE (per request)
┌──────────────────────────────────────────┐   ┌─────────────────────────────────────────┐
│ 01_trust_scorer notebook                 │   │ Mimir — Databricks App (Streamlit)       │
│  · reads 10,088-facility Delta table     │   │  · reads facility_verdicts               │
│  · Tier-1 deterministic cross-field      │   │  · ranked list · India trust map         │
│    scorer over all 18,100 claims         │──▶│  · evidence panels · ranking explainer   │
│  · writes facility_verdicts (Delta)      │   │  · ✨ Tier-2 LLM citation receipts        │
│  · logs run to MLflow                    │   │    (Databricks FM serving, cached)       │
└──────────────────────────────────────────┘   │  · 🔎 semantic search (Vector Search)    │
                                               │  · overrides → Lakebase (Delta fallback) │
                                               └─────────────────────────────────────────┘
```

The two halves meet **only** at the Delta table `workspace.default.facility_verdicts` — the notebook writes it offline, the app reads it live. Changing scoring logic means re-running the notebook; the app is a pure consumer.

**Why deterministic-first:** the ranking must be defensible to a donor or a district health officer. No AI decides trust in Mimir. The LLM's only job is *extracting verbatim citations* from a facility's own records, on demand, cached — and the semantic search is retrieval-only. The app says this out loud in its "How this ranking works — no black box" panel: contradicted first, then evidence tier (0–4 fields), then data completeness. Every facility row shows a one-line "why ranked here."

## Databricks stack (Free Edition)

| Capability | How Mimir uses it |
|---|---|
| **Databricks Apps** | The entire product surface — this repo deploys as the live app |
| **Serverless SQL warehouse** | All Delta reads (verdicts, evidence, receipts cache) |
| **Delta tables** | `facility_verdicts` (18,100 scored claims), `llm_receipts` (citation cache), `planner_overrides` (fallback store) |
| **Foundation Model serving** | `databricks-meta-llama-3-3-70b-instruct` via the app's ambient service-principal auth — no API key for the primary path |
| **Lakebase (Postgres)** | Transactional store for planner overrides/notes; app authenticates as its own service principal via OAuth token minted at call time |
| **Mosaic AI Vector Search** | `facility_desc_index` on `databricks-gte-large-en` embeddings — semantic "describe the care you need" search over facility descriptions |
| **MLflow** | Scorer run logging: parameters, verdict counts, calibration metrics, full verdicts.csv artifact (experiment `facility-trust-scorer`) |

## Engineering decisions & tradeoffs (honest ones)

- **Kill-switch fallbacks everywhere.** Overrides: Lakebase-first with automatic Delta failover — the toast tells the planner which store accepted the write, and the review survives either way. Receipts: Databricks FM serving first, OpenAI `gpt-4o-mini` as except-path fallback so a live demo never dies. A trust tool that loses a planner's field notes has failed at its one job.
- **Lakebase auth, the hard-won detail:** projects-style Lakebase rejects the older `generate_database_credential` flow. Mimir connects with psycopg2 using the app service principal's own OAuth token as the password, minted at call time — zero stored secrets.
- **Free Edition embedding throughput is ~1 row/s**, so the 10k-row Vector Search index syncs in hours, not minutes. We chose full-corpus coverage over a fast subset and built the app to degrade gracefully (a clear "index not available yet" note) while the sync completes. Semantic search is additive by design — verdicts never depend on it.
- **Map honesty:** facilities with impossible coordinates are excluded from the India map and *counted on screen* ("N records excluded — flagged, not plotted") rather than silently dropped. Duplicate city-centroid coordinates get a small deterministic jitter so dense cities (Delhi, Chennai) render as distinct facilities instead of one blob.
- **MLflow scope:** batch run-logging of the scorer (parameters → metrics → artifacts) rather than per-click tracing — the scorer is where provenance matters most; in-app tracing is the natural next step.

## Replicate it

1. **Data:** add the Virtue Foundation dataset share to your workspace (`databricks_virtue_foundation_dataset_dais_2026...facilities`, 10,088 rows).
2. **Score:** run `01_trust_scorer` top to bottom — writes `workspace.default.facility_verdicts`, logs the MLflow run, and (optional cell) creates the Vector Search endpoint `trust_desk_vs` + index `facility_desc_index`.
3. **Configure:** `cp app.yaml.example app.yaml` and fill in your values (`DATABRICKS_WAREHOUSE_ID` resource, optional `OPENAI_API_KEY` for the fallback path, `LAKEBASE_HOST`/`LAKEBASE_INSTANCE` if using Lakebase). `app.yaml` is gitignored — no secrets live in this repo.
4. **Lakebase (optional):** create a Postgres project, add the app's service principal as a role with OAuth; the app creates its own `planner_overrides` table on first write. Without it, everything still works via the Delta fallback.
5. **Deploy:** create a Databricks App from this folder. Grant the app's service principal SELECT on the tables above (and on the Vector Search index + endpoint if using semantic search).
6. Open the app: pick a capability (ICU, maternity, emergency, oncology, trauma, NICU) and a region → verdicts, evidence, receipts, map, overrides.

## What "beyond minimum" looks like here

Minimum workflow (select → ranked list → inspect citations → override with note) ✅ — plus: India trust map with data-desert/medical-desert separation (crisis-mapping stretch), self-correction validator story with logged calibration proof, AI citation receipts grounded in the facility's own text, semantic discovery via Vector Search, dual-store persistence, and a ranking-transparency panel. All live in one deployable app on Free Edition.

---

*Mimir · verdicts from a deterministic evidence scorer · AI extracts citations, never decides trust.*
