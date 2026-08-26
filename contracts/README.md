# GuardClaim — Contracts Notes (Person A)

## twin_confidence divisor decision (Fix A-1)
`twin_confidence = max(0.0, min(1.0, days_of_history / 61))`

Divisor chosen: **61**. This ensures `twin_confidence` reaches exactly 1.0
on the same day `normalization_baseline` transitions to `"personal"` (day 61),
so the two fields are always consistent — no one-day mismatch window.

## D2 boundary note (Fix A-3)
Person A computes `d2_garage_concentration` as a single scalar Herfindahl
index per officer, inside `OfficerTwin`. This is NOT a graph.

Person B builds a separate NetworkX bipartite graph across all officers
and garages for ring detection. That graph does not read or duplicate
this field, and this field does not attempt any cross-officer network
analysis.

**Node format — TO CONFIRM WITH B:** officer_id format is `OFF-XXXXXX`
(hyphen, not underscore). Confirm with B whether their NetworkX graph
nodes use this exact format, so `get_ring_id(officer_id)` lookups succeed
without translation. Update this note once confirmed.

## Synthetic data files (Fix A-2)

**data_utils/raw_data/synthetic_officers.csv** — 600,000 rows (10,000 officers x 60 days)
Columns: officer_id, day, date, d1_approval_rate, d2_garage_concentration,
d3_threshold_proximity, d4_surveyor_diversity, d5_turnaround_variance,
d6_after_hours_pct, archetype, collusion_start_day

**data_utils/raw_data/synthetic_claims.csv** — ~108,000 rows
Columns: claim_id, officer_id, claim_date, claim_amount, garage_id
This is the table Person C's compute_financial_exposure() reads from.