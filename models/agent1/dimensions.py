import numpy as np
import pandas as pd


def compute_d1_approval_rate(officer_claims: pd.DataFrame) -> float:
    """
    D1 - Approval Rate: claims approved / total claims in the window.

    officer_claims: DataFrame with a boolean/int 'approved' column
    (1 = approved, 0 = not approved), for one officer's claims in
    the rolling window.

    Returns the approval rate as a float in [0, 1].
    If the window has zero claims, returns 0.0 (no claims = nothing to approve).
    """
    if len(officer_claims) == 0:
        return 0.0
    return officer_claims["approved"].mean()


def compute_d2_garage_concentration(officer_claims: pd.DataFrame) -> float:
    """
    D2 - Garage Concentration: Herfindahl-Hirschman Index (HHI) of how
    concentrated this officer's claims are across garages.

    officer_claims: DataFrame with a 'garage_id' column, for one officer's
    claims in the rolling window.

    Returns HHI in [0, 1]. Higher = more concentrated (potentially collusive).
    0 claims -> returns 0.0 (no data, assume not concentrated).
    """
    if len(officer_claims) == 0:
        return 0.0

    garage_counts = officer_claims["garage_id"].value_counts()
    garage_shares = garage_counts / len(officer_claims)
    hhi = (garage_shares ** 2).sum()
    return float(hhi)



def compute_d3_threshold_proximity(officer_claims: pd.DataFrame, threshold: float) -> float:
    """
    D3 - Threshold Proximity: % of claims kept just under an escalation
    threshold, to avoid senior review.

    officer_claims: DataFrame with a 'claim_amount' column.
    threshold: the escalation amount (e.g. claim_threshold_75th from
    extract_population_stats()) — passed in, not hardcoded, so different
    datasets/contexts can use different thresholds.

    A claim counts as "threshold-hugging" if it falls in the range
    [threshold * 0.95, threshold) -- i.e. within 5% below the threshold,
    but not over it.

    Returns the proportion of claims that are threshold-hugging, in [0, 1].
    """
    if len(officer_claims) == 0:
        return 0.0

    lower_bound = threshold * 0.95
    is_hugging = (officer_claims["claim_amount"] >= lower_bound) & (officer_claims["claim_amount"] < threshold)
    return float(is_hugging.mean())

def compute_d4_surveyor_diversity(officer_claims: pd.DataFrame) -> float:
    """
    D4 - Surveyor Diversity: unique surveyors used, normalized per 50 claims.

    officer_claims: DataFrame with a 'surveyor_id' column.

    Formula: (unique surveyor count / total claims) * 50
    This normalizes so an officer with 5 claims and 5 surveyors (perfect
    diversity) gets the same score as one with 100 claims and 100 unique
    surveyors -- both are "50 unique surveyors per 50 claims" = 50.0.

    Returns a float >= 0. Higher = more diverse (healthier). Lower =
    more collapsed onto a few surveyors (potential collusion signal).
    0 claims -> returns 0.0.
    """
    if len(officer_claims) == 0:
        return 0.0

    unique_surveyors = officer_claims["surveyor_id"].nunique()
    total_claims = len(officer_claims)
    return (unique_surveyors / total_claims) * 50


def compute_d5_turnaround_variance(officer_claims: pd.DataFrame) -> float:
    """
    D5 - Turnaround Variance: standard deviation of claim processing time.

    officer_claims: DataFrame with a 'processing_days' column (synthetic,
    generated per claim -- see Fix from guide review, NOT derived from
    incident_date - policy_bind_date, which is policy age).

    Returns the standard deviation in days. Higher = more inconsistent
    processing speed (potential collusion signal -- some claims rushed).
    Fewer than 2 claims -> returns 0.0 (can't compute variance from 0-1 points).
    """
    if len(officer_claims) < 2:
        return 0.0

    return float(officer_claims["processing_days"].std())

def compute_d6_after_hours_pct(officer_claims: pd.DataFrame) -> float:
    """
    D6 - After-Hours Activity: % of approvals outside working hours (9 AM-6 PM).

    officer_claims: DataFrame with an 'approval_hour' column (int, 0-23,
    the hour of day the claim was approved).

    A claim counts as after-hours if approval_hour < 9 OR approval_hour >= 18.

    Returns proportion in [0, 1]. Higher = more after-hours activity
    (potential collusion signal -- avoiding supervision).
    0 claims -> returns 0.0.
    """
    if len(officer_claims) == 0:
        return 0.0

    is_after_hours = (officer_claims["approval_hour"] < 9) | (officer_claims["approval_hour"] >= 18)
    return float(is_after_hours.mean())