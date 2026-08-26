# Agent 2: Anomaly Detection Engine (Person B)

## Overview & Mathematical Architecture
Agent 2 operates as the statistical and neural anomaly filter within GuardClaim:
1. **Personal Z-Score Baseline Engine:** Evaluates an officer's raw $D_1$–$D_6$ metrics against their personal historical 60-day moving window ($\mu_{personal}, \sigma_{personal}$) rather than static population averages.
2. **Symmetric Bottleneck Autoencoder ($6 \to 4 \to 2 \to 4 \to 6$):** Ingests globally normalized `_z` vectors from Person A, mapping reconstruction error to continuous percentile ranks [0.0 – 1.0].
3. **Sustained Anomaly Filter:** Uses a 7-day rolling bitmask to suppress transient spikes, escalating to Agent 3 only when anomalies persist for $\ge 5$ of the last 7 calendar days.
4. **Bipartite Collusion Ring Detector:** Evaluates claims routed across repair garages using NetworkX, identifying clusters sharing $\ge 60\%$ of their top-10 dominant garages and assigning persistent `RING-XXXXXX` IDs.
5. **Macro Regional Surge Filter:** Detects regional catastrophe spikes ($\ge 30\%$ regional anomaly rate) and inflates the AE cutoff by $1.20\times$ for 30 days.

## Team Interface Boundaries
* **Person A Interface:** Agent 2 consumes `OfficerTwin`. `PersonalZScoreTracker` reads raw fields (`d1_approval_rate`, etc.), while `AutoencoderScorer` evaluates normalized fields (`d1_approval_rate_z`, etc.).
* **Person C Interface:** Emits strictly validated `AnomalyFlag` instances with continuous AE error, per-dimension Z-scores, sustained duration count, and persistent `network_ring_id`.
* **Person D Interface:** `PersonalBaselineStore` handles in-memory rolling statistics for fast execution; Person D persists snapshots to SQLite `officer_stats`. `get_ring_graph()` exports active subgraphs for evidence PDF generation.