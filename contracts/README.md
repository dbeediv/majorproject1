# GuardClaim — Contracts Notes (Person A)

## twin_confidence divisor decision (Fix A-1)
`twin_confidence = max(0.0, min(1.0, days_of_history / 61))`

Divisor chosen: **61**. This ensures `twin_confidence` reaches exactly 1.0
on the same day `normalization_baseline` transitions to `"personal"` (day 61),
so the two fields are always consistent — no one-day mismatch window.
Verified with an automated regression test (`test_officer_twin_pipeline.py`)
after a bug was found and fixed where confidence stalled at 0.9836 instead
of reaching 1.0 — see full write-up in the progress report if needed.

## D2 boundary note (Fix A-3)
Person A computes `d2_garage_concentration` as a single scalar Herfindahl
index per officer, inside `OfficerTwin`. This is NOT a graph.

Person B builds a separate NetworkX bipartite graph across all officers
and garages for ring detection. That graph does not read or duplicate
this field, and this field does not attempt any cross-officer network
analysis.

**Node format — STILL TO CONFIRM WITH B:** officer_id format is `OFF-XXXXXX`
(hyphen, not underscore). Confirm with B whether their NetworkX graph
nodes use this exact format, so `get_ring_id(officer_id)` lookups succeed
without translation. Update this note once confirmed.

## Synthetic data files (Fix A-2)

**data_utils/raw_data/synthetic_officers.csv** — 600,000 rows (10,000 officers x 60 days)
Columns: officer_id, day, date, d1_approval_rate, d2_garage_concentration,
d3_threshold_proximity, d4_surveyor_diversity, d5_turnaround_variance,
d6_after_hours_pct, archetype, collusion_start_day
This file contains RAW D1-D6 values only, by design — it is the base
synthetic dataset, not the enriched output. See the "final status" note
below for where the z-scored/enriched version lives.

**data_utils/raw_data/synthetic_claims.csv** — ~108,000 rows
Columns: claim_id, officer_id, claim_date, claim_amount, garage_id,
processing_days, approval_hour
This is the table Person C's compute_financial_exposure() reads from.

## Guide review notes (post Stage 2)

### D2 (garage concentration) proxy — Kaggle data only
The Kaggle dataset (arpan129/insurance-fraud-detection) has no garage_id
column. When extract_features() is built in Stage 7 to process real
incoming claims, "garage" is defined as the combination of
auto_make + incident_location. Two claims share a "garage" if both
fields match. This proxy is Kaggle-data-specific — the synthetic
generator (Stage 2) already uses real Faker-based garage IDs directly
and does not need this proxy.

### D3 (threshold proximity) — escalation threshold definition
Escalation threshold = 75th percentile of total_claim_amount from the
Kaggle dataset (**₹70,592.50**). Claims within 5% below this value count
as threshold-hugging. This value is computed once and stored as
claim_threshold_75th in extract_population_stats() (see loader.py).

### D5 (turnaround variance) — processing_days is synthetic, not derived
(incident_date - policy_bind_date) is POLICY AGE, not claim processing
time — these are not the same thing and must not be confused. A separate
synthetic processing_days field is generated per claim during officer
assignment: drawn from a normal distribution (mean=7, std=3, min=1 day).
This is what D5's turnaround variance is actually computed from.

### Corrected dataset figures (supersedes earlier Lock 1 assumptions)
Kaggle dataset (arpan129): 1,000 rows, 24.7% fraud rate (247 fraud /
753 clean) — NOT 15,420 rows / ~6% fraud rate as stated in the original
Lock 1 report (that report referenced a different, now-unavailable
dataset). This changes class imbalance handling for Person C's XGBoost:
scale_pos_weight ≈ 753/247 ≈ 3.05, much lower than typical rare-fraud
ratios — flagged to Person C directly.

## synthetic_officers_scored.csv and OfficerTwin — final status (UPDATED, all stages complete)

The base `synthetic_officers.csv` intentionally stays RAW-only (D1-D6
values, archetype, collusion_start_day) — it is not meant to be the
enriched output.

The FULL 19-column spec (raw D1-D6 + _z columns + twin_confidence +
days_of_history + normalization_baseline) now EXISTS, in two places:

- **`synthetic_officers_scored.csv`** — all 600,000 rows, full columns,
  including population z-scores and Isolation Forest anomaly scores.
  NOT in this GitHub repo — see "Large file note" below for why and how
  to get it.
- **`data_utils/raw_data/officertwin_sample_batch.json`** — a real,
  schema-validated 60-day OfficerTwin batch for one officer, in the
  exact final schema format (contracts/__init__.py). This IS in the repo
  and safe to use as a reference for exact field names/types/ranges.

**B and C can now safely build against `_z` columns, `twin_confidence`,
`days_of_history`, and `normalization_baseline`** — these are no longer
placeholder/future fields, they are computed and verified with a passing
automated test suite (`models/agent1/test_officer_twin_pipeline.py`,
7/7 tests passing).

## Large file note — synthetic_officers_scored.csv

This file (~165 MB, with z-scores + Isolation Forest scores) is TOO LARGE
for GitHub's 100MB limit and is NOT in this repository.

To get it:
- Ask Person A directly for the file (shared via Drive/WhatsApp), OR
- Regenerate it yourself: pull this repo, then run
  `python models/agent1/isolation_forest.py`
  This reads the smaller synthetic_officers.csv (which IS in the repo)
  and produces an identical scored file locally (same random_state=42,
  fully reproducible).