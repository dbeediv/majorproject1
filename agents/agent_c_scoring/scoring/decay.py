# decay.py
# Person C — Fix C-3: update_last_clean_date()
#
# Called by D's daily job orchestrator IMMEDIATELY AFTER B's AnomalyFlag is
# computed, and BEFORE the weekly decay job ever runs (mandatory ordering,
# Fix D-5). Ties decay's clean_days calculation to B's real-time hard_flag
# signal — without this function, last_clean_date never gets set and decay
# has no idea how long an officer has actually been behaving.
#
# Logic:
#   hard_flag_today == False -> last_clean_date = today (streak continues/starts)
#   hard_flag_today == True  -> last_clean_date = None  (streak resets)

from datetime import date

# --- In-memory stand-in for D's real database table -------------------------
# D hosts the real officer_scores table (SQLAlchemy + SQLite/Postgres).
# This dict simulates one column of that table so the function's logic can
# be built and tested right now, without waiting for D's DB layer.
# Swap this out the moment D's db_session is available (see the commented
# real-DB version below).
_OFFICER_CLEAN_DATES: dict[str, date | None] = {}


def update_last_clean_date(officer_id: str, hard_flag_today: bool, today: date,
                            db_session=None) -> date | None:
    """
    Updates and returns the officer's last_clean_date.

    In production, D passes a real db_session (SQLAlchemy) and this writes
    to the officer_scores table. Until that's wired up, it falls back to an
    in-memory dict so the logic can be tested standalone.
    """
    if hard_flag_today:
        new_date = None  # streak resets
    else:
        existing = _OFFICER_CLEAN_DATES.get(officer_id)
        # Preserve the streak's start date — only set to today if there is
        # no clean streak already in progress (existing is None, meaning
        # either never scored before or was flagged yesterday).
        new_date = existing if existing is not None else today

    if db_session is not None:
        # --- REAL DB VERSION (uncomment once D's session is available) ---
        # db_session.execute(
        #     'UPDATE officer_scores SET last_clean_date = :d WHERE officer_id = :oid',
        #     {'d': new_date, 'oid': officer_id}
        # )
        # db_session.commit()
        raise NotImplementedError(
            "db_session provided but real DB write not yet wired up — "
            "uncomment the SQL block above once D's officer_scores table exists."
        )

    _OFFICER_CLEAN_DATES[officer_id] = new_date
    return new_date


def get_last_clean_date(officer_id: str) -> date | None:
    """Read-only lookup — used by decay to compute clean_days. Mirrors what
    D's real DB read would return."""
    return _OFFICER_CLEAN_DATES.get(officer_id)


def clean_days(officer_id: str, today: date) -> int:
    """How many days an officer has been clean, based on last_clean_date.
    Returns 0 if the officer has no clean_date on record (either never
    scored, or currently hard-flagged)."""
    last_clean = get_last_clean_date(officer_id)
    if last_clean is None:
        return 0
    return (today - last_clean).days


if __name__ == "__main__":
    from datetime import timedelta

    print("=== update_last_clean_date() — simulated 10-day run for one officer ===\n")

    officer = "OFF-000001"
    start = date(2026, 8, 1)

    # Simulated daily hard_flag values from B: clean for a while, then
    # flagged for 2 days (streak resets), then clean again.
    simulated_hard_flags = [False, False, False, True, True, False, False, False, False, False]

    for i, flag in enumerate(simulated_hard_flags):
        today = start + timedelta(days=i)
        new_date = update_last_clean_date(officer, hard_flag_today=flag, today=today)
        days_clean = clean_days(officer, today)
        print(f"Day {i+1} ({today}): hard_flag={flag!s:5s} "
              f"-> last_clean_date={new_date}  clean_days={days_clean}")

    print("\nExpected: streak resets to 0 on days 4-5 (flagged), "
          "then rebuilds 1,2,3,4,5 on days 6-10.")

    print("\n=== Second officer, always clean (sanity check) ===")
    officer2 = "OFF-000042"
    for i in range(5):
        today = start + timedelta(days=i)
        update_last_clean_date(officer2, hard_flag_today=False, today=today)
    print(f"OFF-000042 clean_days after 5 clean days: {clean_days(officer2, start + timedelta(days=4))}")
    assert clean_days(officer2, start + timedelta(days=4)) == 4
    print("Assertion passed: 5 clean days starting day 1 -> 4 days elapsed since first clean date.")
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from tier_rules import is_decay_paused, compute_score_band, _CONFIG


def apply_decay_with_band(score: int, band: str, clean_days: int) -> tuple[int, str]:
    """
    Fix C-2: applies weekly decay to a score based on clean_days, then
    recomputes the band via tier_rules.compute_score_band() so bands and
    scores never drift apart.

    Reads points_per_clean_day / max_points_per_run / min_clean_days_before_decay
    straight from tier_config.yaml (single source of truth).

    Returns (new_score, new_band). If the band has decay_paused=True
    (high_risk, confirmed) or clean_days has not met the band's gating
    threshold yet, returns the score/band unchanged.
    """
    if is_decay_paused(band):
        return score, band

    de_esc = _CONFIG["de_escalation"].get(band, {})
    min_days = de_esc.get("min_clean_days_before_decay", 0)
    if clean_days < min_days:
        return score, band

    points_per_day = _CONFIG["decay"]["points_per_clean_day"]
    max_points = _CONFIG["decay"]["max_points_per_run"]

    decay_amount = min(clean_days * points_per_day, max_points)
    new_score = max(0, score - decay_amount)
    new_band = compute_score_band(new_score)

    return new_score, new_band
